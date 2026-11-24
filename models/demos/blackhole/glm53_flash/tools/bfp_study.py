"""How much BFP8 / BFP4 weights lose, and whether reordering features first helps (CPU only, no device).

TT block floating point shares one exponent among 16 values: one row of a 16 x 16 tile face, i.e. 16 consecutive
output features of a ttnn weight [in, out] for one input feature.  `bfp_round` reproduces the host packer
(tt_metal/impl/data_format/blockfloat_common.cpp: the group's largest float exponent is shared, each mantissa with
its hidden bit is shifted right by the exponent difference, then rounded to nearest-even to 7 (BFP8) or 3 (BFP4)
bits and saturated); it is to be checked bit for bit against ttnn's own conversion on a device.

Reordering is exact for the model: the expert/MLP intermediate features can be permuted (gate and up rows and down
columns together; the clamped SwiGLU is elementwise), and the hidden features can be permuted model-wide.  Only the
grouping of each matrix changes, so a matrix is quantized in the permuted order and returned in the original order.

    python -m models.demos.blackhole.glm53_flash.tools.bfp_study weights      # per-matrix errors on a sample of layers
    python -m models.demos.blackhole.glm53_flash.tools.bfp_study end-to-end   # KLD / top-1 against bf16, experts only

The end-to-end study teacher-forces the greedy records (tools/acceptance/greedy-v1, prompt + 64 generated tokens) and
scores the positions that predict the generated tokens.  Its bf16 logits (--out, default data_dir("bfp-study-v1"):
bf16_logits.pt) are the CPU prefill-path reference the device accuracy tests read.
"""

from __future__ import annotations

import argparse
import re
import sys

import torch

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint
from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.paths import data_dir

GROUP = 16
MANTISSA_BITS = {"bfp8": 7, "bfp4": 3}


def bfp_round_exact(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Quantize-dequantize along the last dim in groups of 16, bit for bit as TT's host packer does (fp32 result)."""
    width = MANTISSA_BITS[fmt]
    shape = x.shape
    bits = x.float().reshape(-1, GROUP).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign = bits >> 31
    exp = (bits >> 23) & 0xFF
    mant = (bits & 0x7FFFFF) | (1 << 23)
    zero = exp == 0  # zeros and denormals pack to 0
    shared = torch.where(zero, torch.zeros_like(exp), exp).amax(dim=-1, keepdim=True)
    diff = (shared - exp).clamp(min=0)
    mant = torch.where(diff >= 24, torch.zeros_like(mant), mant >> diff.clamp(max=23))
    shift = 24 - width
    round_bits = mant & ((1 << shift) - 1)
    mant = mant >> shift
    tie = 1 << (shift - 1)
    up = (round_bits > tie) | ((round_bits == tie) & ((mant & 1) == 1))
    mant = torch.clamp(mant + up.to(mant.dtype), max=(1 << width) - 1)
    mant = torch.where(zero, torch.zeros_like(mant), mant)
    value = mant.double() * torch.pow(2.0, (shared - 127 - (width - 1)).double())
    value = torch.where(sign == 1, -value, value)
    return value.float().reshape(shape)


def bfp_round(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """The packer's result in float arithmetic: step 2^(e - (bits - 1)) for the group's largest exponent e, round half
    to even, saturate.  Equal to `bfp_round_exact` for bf16 inputs (the bits the packer drops before rounding are zero
    for 8 significant bits unless the value rounds to 0 anyway); about 10x faster."""
    bits = MANTISSA_BITS[fmt]
    g = x.float().reshape(-1, GROUP)
    amax = g.abs().amax(dim=-1, keepdim=True)
    e = torch.frexp(amax).exponent.to(torch.float32) - 1  # floor(log2(amax)) for amax > 0
    step = torch.pow(2.0, e - (bits - 1))
    lim = float(2**bits - 1)
    q = torch.clamp(torch.round(g / step), -lim, lim) * step
    q = torch.where(amax > 0, q, torch.zeros_like(q))
    return q.reshape(x.shape)


def bfp_round_best(x: torch.Tensor, fmt: str, shifts=(0, 1, 2), weight: torch.Tensor | None = None) -> torch.Tensor:
    """BFP with each group's shared exponent chosen for the least squared error among the largest value's exponent e
    and e - 1, e - 2 (`shifts`): with e - s the group is first clipped to the largest magnitude that exponent holds,
    (2^bits - 1) 2^(e - s - bits + 1) (exact in bf16), so the packer, which takes the exponent from the group's
    largest value, picks e - s itself; the largest values lose a little, the others gain resolution.  A host-side
    step before the packer (the device format unchanged).  fmt "bfp4" / "bfp8" -> the packed values (fp32).
    `weight` (broadcast to x): the squared errors weighted, e.g. by the inputs' mean squares when a group runs along
    the inputs (an imatrix)."""
    bits = MANTISSA_BITS[fmt]
    g = x.float().reshape(-1, GROUP)
    wt = None if weight is None else weight.float().expand_as(x).reshape(-1, GROUP)
    amax = g.abs().amax(dim=-1, keepdim=True)
    e = torch.frexp(amax).exponent.to(torch.float32) - 1
    best_q, best_err = None, None
    for shift in shifts:
        lim = (2**bits - 1) * torch.pow(2.0, e - shift - (bits - 1))
        q = bfp_round(torch.maximum(torch.minimum(g, lim), -lim), fmt)
        err = ((q - g).pow(2) * (1.0 if wt is None else wt)).sum(-1, keepdim=True)
        if best_q is None:
            best_q, best_err = q, err
        else:
            better = err < best_err
            best_q = torch.where(better, q, best_q)
            best_err = torch.where(better, err, best_err)
    return best_q.reshape(x.shape)


def bfp_rounder(fmt: str):
    """fmt "bfp4" / "bfp8" (the packer) or "bfp4-best" / "bfp8-best" (`bfp_round_best`) -> (round(x), base fmt)."""
    if fmt.endswith("-best"):
        base = fmt[: -len("-best")]
        return (lambda x: bfp_round_best(x, base)), base
    return (lambda x: bfp_round(x, fmt)), fmt


FP8_BLOCK = 128
# The tensors the official FP8 checkpoint (zai-org/GLM-5.3-Flash @ 46a050bec130) stores as e4m3 with 128x128 scales,
# 37,338 of them: every language-model MLP projection (routed experts, dense, shared) and the DSA layers' q_a / q_b /
# kv_a / o projections (not the KDA layers' o_proj, nor the vision tower).
FP8_MLP = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.(experts\.\d+\.|shared_experts\.)?(gate|up|down)_proj\.weight$"
)
FP8_DSA = re.compile(
    r"^(model\.language_model\.layers\.\d+\.self_attn\.)(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)\.weight$"
)


def is_fp8(name: str, weight_map) -> bool:
    if FP8_MLP.match(name):
        return True
    m = FP8_DSA.match(name)
    return bool(m) and (m.group(1) + "q_a_proj.weight") in weight_map  # o_proj only in the DSA layers


def fp8_round(w: torch.Tensor) -> torch.Tensor:
    """The official FP8 checkpoint's value of a bf16 weight: e4m3(w / s) * s with s = 128x128 block amax / 448 (fp32;
    the codes match the published checkpoint for 100 % of the values checked)."""
    w = w.float()
    rows, cols = w.shape
    blocks = w.unflatten(0, (-1, FP8_BLOCK)).unflatten(2, (-1, FP8_BLOCK))
    s = blocks.abs().amax(dim=(1, 3), keepdim=True) / 448.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    return ((blocks / s).to(torch.float8_e4m3fn).float() * s).reshape(rows, cols)


def quantize_weight(
    w: torch.Tensor, fmt: str, out_order: torch.Tensor | None = None, along: str = "out"
) -> torch.Tensor:
    """A torch weight [out, in] through TT BFP, its 16-groups along the output features (the ttnn [in, out] layout),
    or along the input features with along="in"; `out_order` permutes the output features first."""
    w = w.float()
    rnd, _ = bfp_rounder(fmt)
    if along == "in":
        return rnd(w)
    if out_order is None:
        return rnd(w.T.contiguous()).T
    q = rnd(w[out_order].T.contiguous()).T
    out = torch.empty_like(q)
    out[out_order] = q
    return out


def rel_err(q: torch.Tensor, w: torch.Tensor) -> float:
    w = w.float()
    return ((q - w).norm() / w.norm()).item()


def row_scale(*ws: torch.Tensor) -> torch.Tensor:
    """Per-output-feature RMS, combined over matrices that must share one order."""
    return torch.stack([w.float().pow(2).mean(dim=1) for w in ws]).sum(dim=0).sqrt()


def study_weights(checkpoint: Glm53Checkpoint, config: Glm53Config, layers, experts) -> None:
    p = "model.language_model.layers.{}.mlp."
    # one model-wide hidden order for every down projection: the hidden features sorted by their RMS over all sampled
    # experts' down rows
    downs = [checkpoint.tensor(p.format(l) + f"experts.{e}.down_proj.weight") for l in layers for e in experts]
    hidden_order = torch.argsort(row_scale(*downs))
    rows = []
    for layer in layers:
        for e in experts:
            prefix = p.format(layer) + f"experts.{e}."
            gate, up, down = (checkpoint.tensor(prefix + f"{n}_proj.weight") for n in ("gate", "up", "down"))
            inter_order = torch.argsort(row_scale(gate, up))
            for fmt in ("bfp8", "bfp4"):
                rows.append(
                    (
                        layer,
                        e,
                        fmt,
                        rel_err(quantize_weight(gate, fmt), gate),
                        rel_err(quantize_weight(gate, fmt, inter_order), gate),
                        rel_err(quantize_weight(gate, fmt, along="in"), gate),
                        rel_err(quantize_weight(up, fmt), up),
                        rel_err(quantize_weight(up, fmt, inter_order), up),
                        rel_err(quantize_weight(down, fmt), down),
                        rel_err(quantize_weight(down, fmt, hidden_order), down),
                        rel_err(quantize_weight(down, fmt, along="in"), down),
                    )
                )
        print(f"layer {layer} done", flush=True)
    head = "gate: natural / inter-sorted / groups-on-in | up: natural / inter-sorted | down: natural / hidden-sorted / groups-on-in"
    for fmt in ("bfp8", "bfp4"):
        sel = torch.tensor([r[3:] for r in rows if r[2] == fmt])
        mean = sel.mean(dim=0).tolist()
        print(f"{fmt} mean rel L2 error over {len(sel)} experts ({head}):")
        print("   " + "  ".join(f"{v:.4%}" for v in mean))
    for fmt in ("bfp8", "bfp4"):
        for layer in layers:
            sel = torch.tensor([r[3:] for r in rows if r[2] == fmt and r[0] == layer]).mean(dim=0).tolist()
            print(f"  {fmt} layer {layer}: " + "  ".join(f"{v:.4%}" for v in sel))


class QuantizedExperts:
    """A checkpoint view whose routed-expert weights come back BFP-rounded (as bf16, which holds them exactly).

    order="natural" keeps the stored order; order="sorted" reorders the intermediate features of each expert by the
    combined RMS of its gate and up rows (one order for both) and the hidden features model-wide by `hidden_order`
    (the down projections' output groups).  Results are cached per layer; `clear` drops them.
    """

    def __init__(
        self, base: Glm53Checkpoint, fmt: str | dict | None, order: str = "natural", hidden_order=None
    ) -> None:
        self.base, self.fmt, self.order, self.hidden_order = base, fmt, order, hidden_order
        self.root, self.weight_map, self.infos = base.root, base.weight_map, base.infos
        self.cache: dict[str, torch.Tensor] = {}

    @property
    def config(self):
        return self.base.config

    def names(self, prefix=""):
        return self.base.names(prefix)

    def clear(self):
        self.cache.clear()

    def tensor(self, name: str) -> torch.Tensor:
        if self.fmt in ("fp8", "fp8-bfp8"):
            return self._fp8(name)
        if self.fmt is None or ".mlp.experts." not in name:
            return self.base.tensor(name)
        if name not in self.cache:
            prefix, proj = name.rsplit(".", 2)[0], name.rsplit(".", 2)[1]
            w = self.base.tensor(name)
            if callable(self.fmt):  # by layer and projection (`layer_formats`)
                fmt = self.fmt(int(name.split(".layers.")[1].split(".")[0]), proj)
            else:
                fmt = self.fmt[proj] if isinstance(self.fmt, dict) else self.fmt
            if self.order == "natural":
                order = None
            elif proj == "down_proj":
                order = self.hidden_order
            else:
                order = torch.argsort(
                    row_scale(
                        self.base.tensor(prefix + ".gate_proj.weight"), self.base.tensor(prefix + ".up_proj.weight")
                    )
                )
            self.cache[name] = quantize_weight(w, fmt, order).to(torch.bfloat16)
        return self.cache[name]


def _fp8_tensor(self, name: str) -> torch.Tensor:
    """fmt "fp8": the FP8 checkpoint's values for its FP8 tensors (bf16 for the rest, as that checkpoint has them);
    fmt "fp8-bfp8": additionally the routed experts BFP8-packed from those values (BFP8 made from the FP8 checkpoint).
    """
    if not is_fp8(name, self.base.weight_map):
        return self.base.tensor(name)
    if name not in self.cache:
        w = fp8_round(self.base.tensor(name))
        if self.fmt == "fp8-bfp8" and ".mlp.experts." in name:
            w = quantize_weight(w, "bfp8")
        self.cache[name] = w.to(torch.bfloat16)
    return self.cache[name]


QuantizedExperts._fp8 = _fp8_tensor


def teacher_forced(checkpoint, config, records, score_slices):
    """Layer-major forward of every record (prefill, no state); the fp32 logits at the scored positions."""
    import torch.nn.functional as F

    from models.demos.blackhole.glm53_flash.tt.common import bf16, rms_norm
    from models.demos.blackhole.glm53_flash.tt.layer import DecoderLayer
    from models.demos.blackhole.glm53_flash.tt.mhc import collapse_streams, expand_streams

    p = "model.language_model."
    embed = bf16(checkpoint.tensor(p + "embed_tokens.weight"))
    streams = [expand_streams(F.embedding(torch.tensor([ids]), embed), config.hc_mult) for ids in records]
    with torch.no_grad():
        for layer in range(config.num_layers):
            module = DecoderLayer(checkpoint, config, layer)
            streams = [module(s)[0] for s in streams]
            if hasattr(checkpoint, "clear"):
                checkpoint.clear()
            print(f"  layer {layer} done", flush=True)
        norm, head = bf16(checkpoint.tensor(p + "norm.weight")), checkpoint.tensor("lm_head.weight").float()
        out = []
        for s, sl in zip(streams, score_slices):
            hidden = rms_norm(collapse_streams(s), norm, config.rms_norm_eps)[0, sl]
            out.append(F.linear(hidden.float(), head))
    return torch.cat(out)


def layer_formats(spec: str):
    """A low-precision pattern as the device's switches spell it, e.g. "bfp4:3-20,30" (GLM53_EXPERT_LAYERS: those
    layers' routed experts in BFP4, the rest BFP8), optionally "+down=bfp4" (GLM53_EXPERT_DOWN_DTYPE: every layer's
    down in that format) -> fmt(layer, proj)."""
    layers_part, _, down = spec.partition("+down=")
    name, _, ranges = layers_part.partition(":")
    chosen = set()
    for part in filter(None, ranges.split(",")):
        lo, _, hi = part.partition("-")
        chosen.update(range(int(lo), int(hi or lo) + 1))

    def fmt(layer: int, proj: str) -> str:
        if proj == "down_proj" and down:
            return down
        return name if layer in chosen else "bfp8"

    return fmt


def topk_router(k: int, renorm: bool):
    """A replacement for tt/moe.Router.__call__ that keeps each token's top `k` of the router's choice (k below the
    model's 8: the fewer-experts trial), their weights normalized over the k (renorm) or over
    the model's top 8 as before (the k kept, the rest dropped)."""
    import torch.nn.functional as F

    from models.demos.blackhole.glm53_flash.tt.moe import Routing

    def call(self, x):
        x = x.reshape(-1, x.shape[-1])
        logits = F.linear(x.float(), self.weight.float())
        scores = logits.sigmoid()
        choice = scores + self.bias
        top = torch.topk(choice, k=self.k, dim=-1, sorted=True)[1]  # the model's top 8, largest first
        indices = top[:, :k]
        weights = scores.gather(1, indices)
        total = (weights if renorm else scores.gather(1, top)).sum(dim=-1, keepdim=True)
        return Routing(logits=logits, indices=indices, weights=weights / (total + 1e-20) * self.scale)

    return call


def end_to_end(checkpoint, config, configs, out_dir) -> None:
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_dir = Path(__file__).with_name("acceptance") / "greedy-v1"
    recs = [json.loads(f.read_text()) for f in sorted(rec_dir.glob("*.json"))]
    seqs = [r["prompt_token_ids"] + r["generated_token_ids"] for r in recs]
    # the positions that predict each generated token
    slices = [
        slice(len(r["prompt_token_ids"]) - 1, len(r["prompt_token_ids"]) - 1 + len(r["generated_token_ids"]))
        for r in recs
    ]
    greedy = torch.tensor([t for r in recs for t in r["generated_token_ids"]])
    base_path = out_dir / "bf16_logits.pt"
    if base_path.exists():
        base = torch.load(base_path)
    else:
        print("bf16 baseline", flush=True)
        base = teacher_forced(checkpoint, config, seqs, slices)
        torch.save(base, base_path)
    print(
        f"bf16 teacher-forced argmax equals the greedy record at {(base.argmax(-1) == greedy).float().mean():.2%} of {len(greedy)} positions",
        flush=True,
    )
    downs = [
        checkpoint.tensor(f"model.language_model.layers.{l}.mlp.experts.{e}.down_proj.weight")
        for l in (3, 20, 44)
        for e in (0, 1, 57, 141, 200, 287)
    ]
    hidden_order = torch.argsort(row_scale(*downs))
    logp_base = torch.log_softmax(base, dim=-1)
    mixes = {"mixed": {"gate_proj": "bfp4", "up_proj": "bfp4", "down_proj": "bfp8"}}  # gate/up BFP4, down BFP8
    from models.demos.blackhole.glm53_flash.tt import moe as tt_moe

    plain_call = tt_moe.Router.__call__
    for full in configs:
        # "<config>+topk=K" / "+topk=Kn": each token's top K experts, renormalized over them / over the top 8 (n)
        name, _, topk = full.partition("+topk=")
        tt_moe.Router.__call__ = topk_router(int(topk.rstrip("n")), not topk.endswith("n")) if topk else plain_call
        if name in ("fp8", "fp8-bfp8"):
            view = QuantizedExperts(checkpoint, name)
        elif name.endswith("-best"):  # every routed expert with the least-error exponents (`bfp_round_best`)
            view = QuantizedExperts(checkpoint, name)
        elif name.startswith("layers="):  # a pattern by layer: layers=bfp4:3-20,30[+down=bfp4]
            view = QuantizedExperts(checkpoint, layer_formats(name[len("layers=") :]))
        else:
            fmt, _, order = name.partition("-")
            view = QuantizedExperts(checkpoint, mixes.get(fmt, fmt), order or "natural", hidden_order)
        print(full, flush=True)
        logits = teacher_forced(view, config, seqs, slices)
        logp = torch.log_softmax(logits, dim=-1)
        kld = (logp_base.exp() * (logp_base - logp)).sum(-1)
        top1 = (logits.argmax(-1) == base.argmax(-1)).float()
        summary = {
            "config": full,
            "positions": len(top1),
            "mean_kld": kld.mean().item(),
            "p99_kld": kld.quantile(0.99).item(),
            "top1_agreement": top1.mean().item(),
            "agreement_with_greedy": (logits.argmax(-1) == greedy).float().mean().item(),
        }
        print(json.dumps(summary), flush=True)
        stem = name if not name.startswith("layers=") else "layers-" + re.sub(r"[^0-9a-z+=,-]", "_", name[7:])
        stem += f"-topk{topk}" if topk else ""
        (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1) + "\n")
    tt_moe.Router.__call__ = plain_call


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    weights = sub.add_parser("weights")
    weights.add_argument("--layers", default="3,20,44")
    weights.add_argument("--experts", default="0,1,57,141,200,287")
    e2e = sub.add_parser("end-to-end")
    e2e.add_argument(
        "--configs",
        default="bfp8,bfp8-sorted,bfp4,bfp4-sorted",
        help="also: mixed (gate/up BFP4, down BFP8), fp8 (the FP8 checkpoint's values), fp8-bfp8 (+ experts BFP8), "
        "layers=bfp4:3-20,30[+down=bfp4] (by layer, as GLM53_EXPERT_LAYERS / GLM53_EXPERT_DOWN_DTYPE; configs are "
        "split at ';' when one has commas); any of them +topk=K or +topk=Kn (each token's top K experts, weights "
        "renormalized over the K, or n: over the model's top 8)",
    )
    e2e.add_argument("--out", default=str(data_dir("bfp-study-v1")))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    checkpoint = Glm53Checkpoint(args.checkpoint)
    config = Glm53Config.from_dict(checkpoint.config)
    if args.command == "weights":
        study_weights(
            checkpoint, config, [int(x) for x in args.layers.split(",")], [int(x) for x in args.experts.split(",")]
        )
    if args.command == "end-to-end":
        semi = ";" in args.configs or "layers=" in args.configs
        configs = args.configs.split(";") if semi else args.configs.split(",")
        end_to_end(checkpoint, config, configs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
