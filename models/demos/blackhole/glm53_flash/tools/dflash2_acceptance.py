"""How many of the DFlash2 drafter's tokens (tt/dflash2.py, canada-quant/GLM-5.3-Flash-DFlash2-G) the target would
accept on the greedy records (tools/acceptance/greedy-v1), next to the MTP layer's (tools/mtp_acceptance.py).

Stage 1, the target (`tt/`, CPU, bf16), once per record and cached in --taps-dir: one teacher-forced pass over the
prompt and the record's tokens, keeping the mean of the 4 mHC streams after the layers the drafter taps (and the layers
before them, for --tap-modes in), the final hidden state, the main head's argmax and the MTP layer's drafts (the first,
and the second chained over the whole sequence as in mtp_acceptance's default).

Stage 2, the drafter: at every bonus position s of the generated part (the bonus token t[s], the context the taps of
positions 0 .. s - 1), the block [t[s], mask x K] gives K drafts; n = how many of them match t[s + 1] .. from the
first.  Per K: the mean n over the positions whose K targets are all inside the record, the survival P(n >= j), the
walk of real speculation (from s = p: accept n, move n + 1, until the window passes the record's end), and the
spread over records (a bootstrap over the records).  K is a separate forward each (the block's attention sees every
mask slot).

    python -m models.demos.blackhole.glm53_flash.tools.dflash2_acceptance [--ids fact,code] [--ks 3,5,7]
        [--tap-modes out,in] [--taps-dir DIR] [--out JSON]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from models.demos.blackhole.glm53_flash import checkpoint as checkpoint_module
from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint
from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.paths import data_dir, drafter_dir
from models.demos.blackhole.glm53_flash.tt.common import rms_norm
from models.demos.blackhole.glm53_flash.tt.dflash2 import DFlash2Drafter
from models.demos.blackhole.glm53_flash.tt.mhc import collapse_streams, expand_streams
from models.demos.blackhole.glm53_flash.tt.model import Glm53TextModel
from models.demos.blackhole.glm53_flash.tt.mtp import MtpLayer

RECORDS = Path(checkpoint_module.__file__).parent / "tools" / "acceptance" / "greedy-v1"
DEFAULT_TAPS = data_dir("dflash2-v1", "taps")


def target_pass(model: Glm53TextModel, mtp: MtpLayer | None, ids: torch.Tensor, keep: set[int]) -> dict:
    """One teacher-forced pass: the stream means after the layers in `keep`, the final hidden state, the main argmax
    and the MTP drafts (position i: the first predicts t[i + 2], the chained second t[i + 3])."""
    cfg = model.config
    after = {}
    with torch.no_grad():
        streams = expand_streams(F.embedding(ids, model.embed), cfg.hc_mult)
        for i, layer in enumerate(model.layers):
            streams, _, _ = layer(streams, None)
            if i in keep:
                after[i] = collapse_streams(streams)[0].clone()
        hidden = rms_norm(collapse_streams(streams), model.norm, cfg.rms_norm_eps)  # [1, L, H]
        out = {"after": after, "final": hidden[0].clone(), "main_top": model.logits(hidden)[0].argmax(-1)}
        if mtp is not None:
            embeds = F.embedding(ids, model.embed)
            y, _ = mtp(embeds[:, 1:], hidden[:, :-1])
            out["mtp1"] = model.logits(mtp.head_input(y))[0].argmax(-1)
            y2, _ = mtp(embeds[:, 2:], y[:, :-1])
            out["mtp2"] = model.logits(mtp.head_input(y2))[0].argmax(-1)
    return out


def taps_for(saved: dict, layers: tuple[int, ...], mode: str) -> torch.Tensor:
    """[L, n_taps * H]: the stream mean after layer l (mode out, HF's hidden_states[l + 1]) or before it (mode in)."""
    shift = 0 if mode == "out" else 1
    return torch.cat([saved["after"][l - shift] for l in layers], dim=-1)


def mtp_walks(mtp_rows: dict) -> dict:
    """The MTP walks (one draft; two, the second chained) from the first generated token, as for the drafter."""
    out = {}
    for k in (1, 2):
        rec = []
        for first, second in mtp_rows.values():
            s, tok, steps = min(first), 0, 0
            while s + k - 1 in first:  # the window's last target t[s + k] is in the record
                n = first[s] + (second.get(s, 0) if k == 2 and first[s] else 0)
                tok, steps, s = tok + n + 1, steps + 1, s + n + 1
            rec.append((tok, steps))
        lo, hi = bootstrap(rec)
        out[f"walk_tokens_per_step_k{k}"] = round(sum(t for t, _ in rec) / sum(n for _, n in rec), 4)
        out[f"walk_ci95_k{k}"] = [round(lo, 3), round(hi, 3)]
    return out


def bootstrap(per_record: list[tuple[float, int]], reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % interval of the pooled mean, resampling records: per_record = [(sum, count)]."""
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        pick = [per_record[rng.randrange(len(per_record))] for _ in per_record]
        n = sum(c for _, c in pick)
        means.append(sum(s for s, _ in pick) / n if n else 0.0)
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", default=None, help="comma-separated record ids (default: all)")
    parser.add_argument("--ks", default="3,5,7", help="the numbers of mask slots (drafts) per block")
    parser.add_argument("--tap-modes", default="out", help="out (after the layer), in (before it), or both")
    parser.add_argument("--drafter", type=Path, default=None, help="the drafter's checkpoint ($GLM53_DFLASH2_DRAFTER)")
    parser.add_argument("--taps-dir", type=Path, default=DEFAULT_TAPS)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="a JSON of the results")
    parser.add_argument(
        "--drafter-fmt",
        default=None,
        help="bfp8 / bfp4: the drafter's matmul weights (--drafter-fmt-keys) rounded as the device's packer does "
        "(tools/bfp_study.bfp_round_exact, groups of 16 output features of the ttnn [in, out] form)",
    )
    parser.add_argument(
        "--drafter-fmt-keys",
        default=r"^(fc|layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj|(attention|mlp)_conv\.kernel_projection))\.weight$",
        help="a regular expression of the weights --drafter-fmt rounds (default: the drafter's own matmuls; not the "
        "head, the embedding or the selector)",
    )
    args = parser.parse_args()
    args.drafter = drafter_dir(args.drafter)
    ks = [int(k) for k in args.ks.split(",")]
    modes = args.tap_modes.split(",")
    records = [json.loads(f.read_text()) for f in sorted(RECORDS.glob("*.json"))]
    if args.ids:
        records = [r for r in records if r["id"] in args.ids.split(",")]
    args.taps_dir.mkdir(parents=True, exist_ok=True)

    # stage 1: the target, cached per record
    t0 = time.time()
    drafter_cfg = json.loads((args.drafter / "config.json").read_text())
    layers = tuple(drafter_cfg["dflash_config"]["target_layer_ids"])
    keep = set(layers) | {l - 1 for l in layers}
    missing = [r for r in records if not (args.taps_dir / f"{r['id']}.pt").exists()]
    if missing:
        checkpoint = Glm53Checkpoint()
        config = Glm53Config.from_dict(checkpoint.config)
        model = Glm53TextModel(checkpoint, config)
        mtp = None if args.no_mtp else MtpLayer(checkpoint, config, "eh")
        for r in missing:
            t1 = time.time()
            ids = torch.tensor([r["prompt_token_ids"] + r["generated_token_ids"]])
            saved = target_pass(model, mtp, ids, keep)
            saved.update(ids=ids[0], prompt=len(r["prompt_token_ids"]))
            torch.save(saved, args.taps_dir / f"{r['id']}.pt")
            print(f"target {r['id']}: {ids.shape[1]} tokens, {time.time() - t1:.0f} s", flush=True)
        del model, mtp
    print(f"stage 1: {time.time() - t0:.0f} s", flush=True)

    # stage 2: the drafter
    t0 = time.time()
    drafter = DFlash2Drafter(args.drafter)
    if args.drafter_fmt:
        import re

        from models.demos.blackhole.glm53_flash.tools.bfp_study import bfp_round_exact

        pick = re.compile(args.drafter_fmt_keys)
        names = sorted(k for k in drafter.w if pick.search(k))
        for k in names:  # torch [out, in] -> ttnn [in, out]: the shared exponent over 16 output features
            drafter.w[k] = bfp_round_exact(drafter.w[k].T.contiguous(), args.drafter_fmt).T.contiguous().bfloat16()
        print(f"drafter weights {args.drafter_fmt}: {len(names)} tensors ({names[0]} .. {names[-1]})", flush=True)
    print(f"drafter loaded: {time.time() - t0:.0f} s", flush=True)
    results = {"ks": ks, "modes": modes, "layers": layers, "records": {}, "summary": {}}
    acc = {}  # (mode, k) -> per record: dict of per-anchor arrays
    mtp_rows = {}
    for r in records:
        saved = torch.load(args.taps_dir / f"{r['id']}.pt", weights_only=False)
        ids, p = saved["ids"], saved["prompt"]
        length = ids.shape[0]
        tokens = ids.tolist()
        same = int((saved["main_top"][p - 1 : length - 1] == ids[p:length]).sum())
        row = {"prompt": p, "generated": length - p, "main_reproduces": f"{same}/{length - p}"}
        if (
            "mtp1" in saved
        ):  # bonus at s (p .. L - 2): first draft mtp1[s - 1] vs t[s + 1]; second mtp2[s - 1] vs t[s + 2]
            first = {s: int(saved["mtp1"][s - 1] == ids[s + 1]) for s in range(p, length - 1)}
            second = {s: int(saved["mtp2"][s - 1] == ids[s + 2]) for s in range(p, length - 2) if first[s]}
            mtp_rows[r["id"]] = (first, second)
        for mode in modes:
            t1 = time.time()
            ctx = drafter.context(taps_for(saved, layers, mode))
            for k in ks:
                starts = torch.arange(p, length - 1)  # every bonus position with at least one target token after it
                bonus = ids[starts]
                with torch.no_grad():
                    hidden = drafter.forward(ctx, starts, bonus, k)
                    ref = torch.full((len(starts), k), -1, dtype=torch.long)
                    for a, s in enumerate(starts.tolist()):
                        avail = min(k, length - 1 - s)
                        ref[a, :avail] = ids[s + 1 : s + 1 + avail]
                    prop = drafter.propose(hidden, bonus, ref)
                per = []
                for a, s in enumerate(starts.tolist()):
                    window = min(k, length - 1 - s)
                    n = 0
                    while n < window and int(prop.drafts[a, n]) == tokens[s + 1 + n]:
                        n += 1
                    n_plain = 0
                    while n_plain < window and int(prop.plain[a, n_plain]) == tokens[s + 1 + n_plain]:
                        n_plain += 1
                    per.append(
                        {
                            "s": s,
                            "window": window,
                            "rest": length - 1 - s,
                            "n": n,
                            "n_plain": n_plain,
                            "top_hit_1": bool(prop.in_top_k[a, 0]),
                            "drafts": prop.drafts[a].tolist(),
                        }
                    )
                acc.setdefault((mode, k), {})[r["id"]] = per
                full = [x for x in per if x["window"] == k]
                mean_n = sum(x["n"] for x in full) / max(len(full), 1)
                row[f"{mode} K={k}"] = {"positions": len(full), "mean_n": round(mean_n, 3)}
                print(
                    f"{r['id']} {mode} K={k}: {len(full)} positions, mean accepted {mean_n:.3f}, slot 1 "
                    f"{sum(x['n'] >= 1 for x in full) / max(len(full), 1):.3f} (plain "
                    f"{sum(x['n_plain'] >= 1 for x in full) / max(len(full), 1):.3f})",
                    flush=True,
                )
            row[f"{mode} seconds"] = round(time.time() - t1)
        if r["id"] in mtp_rows:
            first, second = mtp_rows[r["id"]]
            row["mtp first"] = round(sum(first.values()) / len(first), 3)
            row["mtp second"] = round(sum(second.values()) / max(len(second), 1), 3)
        results["records"][r["id"]] = row
        print(row, flush=True)

    # the summary
    for (mode, k), recs in acc.items():
        full = {rid: [x for x in per if x["window"] == k] for rid, per in recs.items()}
        pooled = [x for per in full.values() for x in per]
        count = len(pooled)
        mean_n = sum(x["n"] for x in pooled) / count
        survival = [sum(x["n"] >= j for x in pooled) / count for j in range(1, k + 1)]
        survival_plain = [sum(x["n_plain"] >= j for x in pooled) / count for j in range(1, k + 1)]
        lo, hi = bootstrap([(sum(x["n"] for x in per), len(per)) for per in full.values() if per])
        per_record = {rid: round(sum(x["n"] for x in per) / len(per), 3) for rid, per in full.items() if per}
        # the walk: from the first generated token, accept n, move n + 1, while the window fits in the record
        walk_tokens, walk_steps, walk_rec, walk_n = 0, 0, [], []
        for rid, per in recs.items():
            by_s = {x["s"]: x for x in per}
            s, tok, steps = min(by_s), 0, 0
            while s in by_s and by_s[s]["window"] == k:
                tok += by_s[s]["n"] + 1
                walk_n.append(by_s[s]["n"])
                steps += 1
                s += by_s[s]["n"] + 1
            walk_tokens += tok
            walk_steps += steps
            walk_rec.append((tok, steps))
        wlo, whi = bootstrap(walk_rec)
        # MTP on the same positions
        mtp_first = [mtp_rows[rid][0][x["s"]] for rid, per in full.items() if rid in mtp_rows for x in per]
        summary = {
            "positions": count,
            "mean_accepted": round(mean_n, 4),
            "tokens_per_step": round(1 + mean_n, 4),
            "ci95_tokens_per_step": [round(1 + lo, 3), round(1 + hi, 3)],
            "survival": [round(v, 4) for v in survival],
            "survival_plain": [round(v, 4) for v in survival_plain],
            "plain_tokens_per_step": round(1 + sum(survival_plain), 4),
            "slot1_target_in_top16": round(sum(x["top_hit_1"] for x in pooled) / count, 4),
            "walk_tokens_per_step": round(walk_tokens / max(walk_steps, 1), 4),
            "walk_steps": walk_steps,
            "walk_survival": [round(sum(n >= j for n in walk_n) / len(walk_n), 4) for j in range(1, k + 1)],
            "walk_ci95": [round(wlo, 3), round(whi, 3)],
            "per_record_mean_accepted": per_record,
            "mtp_first_same_positions": round(sum(mtp_first) / len(mtp_first), 4) if mtp_first else None,
        }
        results["summary"][f"{mode} K={k}"] = summary
        print(f"== {mode} K={k}: {json.dumps(summary)}", flush=True)
    # every K on the positions where the largest K's window fits, and the largest block's drafts cut to K (the walk is
    # left to right, so its first K drafts are those of a K-draft walk over the bigger block)
    kmax = max(ks)
    for mode in modes:
        common = {}
        for k in ks:
            pooled = [x for per in acc[(mode, k)].values() for x in per if x["rest"] >= kmax]
            common[f"K={k} own block"] = round(1 + sum(x["n"] for x in pooled) / len(pooled), 4)
            big = [x for per in acc[(mode, kmax)].values() for x in per if x["window"] == kmax]
            common[f"K={k} cut from K={kmax}"] = round(1 + sum(min(x["n"], k) for x in big) / len(big), 4)
        common["positions"] = len(pooled)
        results["summary"][f"{mode} common positions"] = common
        print(f"== {mode} tokens per step on the K={kmax} positions: {json.dumps(common)}", flush=True)
    if mtp_rows:
        f_all = [v for first, _ in mtp_rows.values() for v in first.values()]
        s_all = [v for _, second in mtp_rows.values() for v in second.values()]
        a1, a2 = sum(f_all) / len(f_all), sum(s_all) / max(len(s_all), 1)
        lo, hi = bootstrap([(sum(first.values()), len(first)) for first, _ in mtp_rows.values()])
        results["summary"]["mtp"] = {
            "first": round(a1, 4),
            "first_ci95": [round(lo, 3), round(hi, 3)],
            "second_given_first": round(a2, 4),
            "positions": len(f_all),
            "tokens_per_step_k1": round(1 + a1, 4),
            "tokens_per_step_k2": round(1 + a1 + a1 * a2, 4),
            **mtp_walks(mtp_rows),
        }
        print(f"== mtp: {json.dumps(results['summary']['mtp'])}", flush=True)
    print(f"stage 2: {time.time() - t0:.0f} s", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results["per_position"] = {f"{m} K={k}": v for (m, k), v in acc.items()}
        args.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
