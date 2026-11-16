"""How often the MTP layer's draft equals the model's own next greedy token (`tt/`, CPU, bf16): the acceptance rate of
MTP self-speculative decode with greedy verification.

The MTP layer is `tt/mtp.py`'s.  Along each greedy record (tools/acceptance/greedy-v1: the prompt and the model's 64
greedy tokens, one teacher-forced pass of the main model) the draft at i is accepted when it equals the record's token
t[i+2].  A second draft, chained (the MTP layer again on its own output hidden state and embed(t[i+2])), is measured
too, given the first was right (teacher-forced).  `--order he` takes the other concatenation order.

    python -m models.demos.blackhole.glm53_flash.tools.mtp_acceptance [--ids fact,json] [--order eh|he]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from models.demos.blackhole.glm53_flash import checkpoint as checkpoint_module
from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint
from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.tt.dsa import DsaState
from models.demos.blackhole.glm53_flash.tt.model import Glm53TextModel
from models.demos.blackhole.glm53_flash.tt.mtp import MtpLayer

RECORDS = Path(checkpoint_module.__file__).parent / "tools" / "acceptance" / "greedy-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", default=None, help="comma-separated record ids (default: all)")
    parser.add_argument("--order", default="eh", choices=("eh", "he"))
    parser.add_argument("--out", type=Path, default=None, help="a JSON of the per-record results")
    parser.add_argument(
        "--chain",
        default="sequence",
        choices=("sequence", "eagle", "single"),
        help="the second draft's context: sequence (the MTP layer over the chained inputs as a whole sequence), "
        "eagle (the first pass's cache up to i, then the chained step itself) or single (the device's verify step "
        "with two drafts, the second level of its slots A, B, C at p .. p + 2 in one call: slot s's chained step "
        "sees the first pass's cache up to i - s, then the earlier slots' chained entries; s = 0 is eagle)",
    )
    args = parser.parse_args()
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    records = [json.loads(f.read_text()) for f in sorted(RECORDS.glob("*.json"))]
    if args.ids:
        records = [r for r in records if r["id"] in args.ids.split(",")]
    model = Glm53TextModel(checkpoint, config)
    mtp = MtpLayer(checkpoint, config, args.order)
    totals = {"first": [0, 0], "second": [0, 0], **{f"second slot {s}": [0, 0] for s in range(3)}}
    rows = []
    for r in records:
        t0 = time.time()
        prompt, gen = r["prompt_token_ids"], r["generated_token_ids"]
        ids = torch.tensor([prompt + gen])
        p, n = len(prompt), len(gen)
        with torch.no_grad():
            hidden, _, _ = model.hidden(ids)  # [1, L, H]
            main_top = model.logits(hidden)[0].argmax(-1)  # position i predicts t[i+1]
            embeds = F.embedding(ids, model.embed)
            y, cache = mtp(embeds[:, 1:], hidden[:, :-1])  # position i: from h[i] and t[i+1], predicts t[i+2]
            draft = model.logits(mtp.head_input(y))[0].argmax(-1)
            if args.chain == "sequence":
                y2, _ = mtp(embeds[:, 2:], y[:, :-1])  # chained: from y[i] and t[i+2], predicts t[i+3]
                draft2 = model.logits(mtp.head_input(y2))[0].argmax(-1)
            elif args.chain == "eagle":  # each chained step on the first pass's cache of positions 0 .. i, at i + 1
                draft2 = torch.zeros_like(draft)
                for i in range(p - 1, p + n - 3):
                    past = DsaState(cache.latent[:, : i + 1], cache.index_key[:, : i + 1], cache.index_gate[:, : i + 1])
                    yi, _ = mtp(embeds[:, i + 2 : i + 3], y[:, i : i + 1], past)
                    draft2[i] = model.logits(mtp.head_input(yi))[0, -1].argmax(-1)
            else:  # single: the chained entries of every position (y[j], embed(draft[j])), their cache rows at j + 1
                _, chained = mtp(F.embedding(draft[None], model.embed), y)
                slots = [torch.zeros_like(draft) for _ in range(3)]
                for s in range(3):
                    for i in range(p - 1 + s, p + n - 3):
                        cut = lambda a, b: torch.cat([a[:, : i - s + 1], b[:, i - s : i]], dim=1)  # noqa: E731
                        past = DsaState(
                            cut(cache.latent, chained.latent),
                            cut(cache.index_key, chained.index_key),
                            cut(cache.index_gate, chained.index_gate),
                        )
                        yi, _ = mtp(embeds[:, i + 2 : i + 3], y[:, i : i + 1], past)
                        slots[s][i] = model.logits(mtp.head_input(yi))[0, -1].argmax(-1)
                draft2 = slots[0]
        tokens = ids[0]
        # the main model reproduces the record (teacher-forced greedy) at the generated positions
        same = int((main_top[p - 1 : p + n - 1] == tokens[p : p + n]).sum())
        # first drafts for the generated tokens t[p + 1 ..]: made at i = p - 1 .. p + n - 3
        first = [int(draft[i] == tokens[i + 2]) for i in range(p - 1, p + n - 2)]
        # the second draft counts only where the first was right (it is verified only then)
        second = [int(draft2[i] == tokens[i + 3]) for i in range(p - 1, p + n - 3) if first[i - (p - 1)]]
        totals["first"][0] += sum(first)
        totals["first"][1] += len(first)
        totals["second"][0] += sum(second)
        totals["second"][1] += len(second)
        if args.chain == "single":  # each slot's second drafts (slot s from i = p - 1 + s on), given the first
            for s in range(3):
                got = [int(slots[s][i] == tokens[i + 3]) for i in range(p - 1 + s, p + n - 3) if first[i - (p - 1)]]
                totals[f"second slot {s}"][0] += sum(got)
                totals[f"second slot {s}"][1] += len(got)
        row = {
            "id": r["id"],
            "prompt": p,
            "main_reproduces": f"{same}/{n}",
            "first": round(sum(first) / len(first), 3),
            "second": round(sum(second) / len(second), 3),
            "seconds": round(time.time() - t0),
        }
        rows.append(row)
        print(row, flush=True)
    a1 = totals["first"][0] / totals["first"][1]
    a2 = totals["second"][0] / totals["second"][1]
    print(
        f"order {args.order}: first draft accepted {a1:.3f} ({totals['first'][1]} positions), second (chained, "
        f"given the first) {a2:.3f}; tokens a step with one draft {1 + a1:.2f}, with two {1 + a1 + a1 * a2:.2f}",
        flush=True,
    )
    if args.chain == "single":
        per = {s: totals[f"second slot {s}"][0] / max(totals[f"second slot {s}"][1], 1) for s in range(3)}
        print(f"second draft given the first, by slot (A exact, B and C in one call): {per}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out = {"order": args.order, "chain": args.chain, "first": a1, "second": a2, "records": rows}
        args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
