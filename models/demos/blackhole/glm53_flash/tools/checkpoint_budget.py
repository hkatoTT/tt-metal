"""Per-device memory budget of GLM-5.3-Flash on a Blackhole Galaxy, from the checkpoint's shard headers (no data read).

    python -m models.demos.blackhole.glm53_flash.tools.checkpoint_budget [--ep 32] [--tp 8] [--expert-bytes 2]

The routed experts are split over `--ep` devices (288 / ep per device per layer); the other weights are split over
`--tp` devices (the 1-D vectors and the mHC projections are replicated) and replicated over the remaining devices.
Byte widths: bf16 2, BFP8 1.0625, BFP4 0.5625 per value.  The per-user state and the per-token KV are printed beside
the weights, so the room left for batch x context can be read off.

The state is counted in its smallest form: the KDA conv state as the last kernel - 1 inputs in bf16 (what a step
reads; tt/ keeps HF's kernel-wide window), and the DSA cache as the 512-wide latent plus the indexer key and gate.
The device version keeps more today: an fp32 conv history, no indexer yet, and the DSA latent on each of a row's 4 chips
(expanded keys and values per head would be 16 times the bytes).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint
from models.demos.blackhole.glm53_flash.config import Glm53Config

GiB = 1 << 30


def category(name: str, num_layers: int) -> str:
    if name.startswith("model.visual."):
        return "vision"
    if name == "lm_head.weight":
        return "lm_head"
    if name.endswith("embed_tokens.weight"):
        return "embedding"
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(.*)", name)
    if not m:
        return "final norm"
    layer, rest = int(m.group(1)), m.group(2)
    if layer >= num_layers:
        return "mtp: experts" if ".experts." in rest else "mtp: other"
    if ".experts." in rest:
        return "routed experts"
    if "shared_experts" in rest:
        return "shared expert"
    if rest.startswith("mlp.gate."):
        return "router"
    if rest.startswith("mlp."):
        return "dense mlp"
    if rest.startswith("hc_"):
        return "mhc"
    if "indexer" in rest:
        return "dsa indexer"
    if rest.startswith("self_attn."):
        return "attention"
    return "norms"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--devices", type=int, default=32)
    parser.add_argument("--ep", type=int, default=32, help="devices the routed experts are split over")
    parser.add_argument("--tp", type=int, default=8, help="devices the other matrices are split over")
    parser.add_argument("--expert-bytes", type=float, default=2.0)
    parser.add_argument("--dense-bytes", type=float, default=2.0)
    parser.add_argument("--dram-gib", type=float, default=32.0, help="DRAM per device")
    parser.add_argument("--mtp", action="store_true", help="count the MTP layer")
    args = parser.parse_args()

    checkpoint = Glm53Checkpoint(args.checkpoint)
    config = Glm53Config.from_dict(checkpoint.config)
    values = defaultdict(int)  # parameter counts
    for info in checkpoint.infos.values():
        values[category(info.name, config.num_layers)] += info.numel

    print(f"{'category':<16} {'params':>14} {'bf16 GiB':>9} {'per device GiB':>15}")
    total_device = 0.0
    for cat in sorted(values, key=lambda c: -values[c]):
        n = values[cat]
        if cat == "vision" or (cat.startswith("mtp") and not args.mtp):
            per_device = 0.0
        elif cat in ("routed experts", "mtp: experts"):
            per_device = n * args.expert_bytes / args.ep
        elif cat in ("mhc", "norms", "final norm", "router"):
            per_device = n * 2.0
        else:
            per_device = n * args.dense_bytes / args.tp
        total_device += per_device
        print(f"{cat:<16} {n:>14,} {n * 2 / GiB:>9.2f} {per_device / GiB:>15.3f}")
    total = sum(values.values())
    print(f"{'total':<16} {total:>14,} {total * 2 / GiB:>9.2f} {total_device / GiB:>15.3f}")

    kda, dsa = len(config.kda_layers), len(config.dsa_layers) + (config.num_mtp_layers if args.mtp else 0)
    kda_state = kda * config.kda_num_heads * config.kda_head_dim**2 * 4
    kda_conv = kda * 3 * config.kda_num_heads * config.kda_head_dim * (config.kda_conv_kernel - 1) * 2
    dsa_token = dsa * (config.kv_lora_rank + 2 * config.index_head_dim) * 2
    free = args.dram_gib * GiB - total_device
    print(
        f"\nper user: KDA state {kda_state / 2**20:.1f} MiB fp32 + conv {kda_conv / 2**20:.2f} MiB;"
        f" per token: DSA latent + indexer {dsa_token / 1024:.1f} KiB (bf16, {dsa} layers)"
    )
    print(f"weights per device {total_device / GiB:.2f} GiB of {args.dram_gib:.0f}; left {free / GiB:.2f} GiB")
    for users, context in [(1, 131072), (32, 8192), (64, 8192), (128, 4096), (256, 4096), (256, 8192)]:
        need = users * (kda_state + kda_conv) + users * context * dsa_token
        print(
            f"  {users:>3} users x {context:>6} tokens: {need / GiB:7.2f} GiB total, {need / args.devices / GiB:6.2f} per device"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
