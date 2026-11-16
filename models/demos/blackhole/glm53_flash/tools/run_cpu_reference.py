"""Greedy records from the `tt/` CPU reference: the token streams the device is accepted against.

`tt/` equals HF transformers' `glm5_next` (bf16 weights, eager experts, sdpa) bit for bit, checked per layer, on HF's
own 4-layer model with the decode cache, and on the full 45 layers (`hf_reference compare-full`); the records are
therefore HF's greedy streams, with the LM head in fp32.  Each step keeps the top 16 logits and the top-1 margin, so a
later divergence can be told apart from a near-tie.

    python -m models.demos.blackhole.glm53_flash.tools.run_cpu_reference --out DIR [--ids fact,json] [--max-new-tokens 64]

One JSON record per prompt (`<out>/<id>.json`); the prompts are `tools/acceptance/prompts.json`.  With
`--teacher-decode $GLM53_DATA_DIR/teacher-decode-v1/bf16_decode_logits.pt` it instead saves the decode-path logits on
the records in tools/acceptance/greedy-v1, the CPU reference the device accuracy tests read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from models.demos.blackhole.glm53_flash.checkpoint import PINNED_REVISION, Glm53Checkpoint
from models.demos.blackhole.glm53_flash.tt.model import Glm53TextModel

PROMPTS = Path(__file__).with_name("acceptance") / "prompts.json"
TOP = 16


def render(tokenizer, prompt: dict) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        prompt["messages"],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        **prompt.get("chat_template_kwargs", {}),
    )
    return [int(i) for i in rendered["input_ids"]]


def greedy(model: Glm53TextModel, prompt_ids: list[int], max_new_tokens: int, eos: set[int]) -> list[dict]:
    steps = []
    state = None
    ids = torch.tensor([prompt_ids])
    for _ in range(max_new_tokens):
        t0 = time.time()
        with torch.no_grad():
            logits, state, _ = model(ids, state)
        last = logits[0, -1]
        top = last.topk(TOP)
        token = int(last.argmax())
        steps.append(
            {
                "token": token,
                "top_ids": [int(i) for i in top.indices],
                "top_logits": [round(float(v), 4) for v in top.values],
                "margin": round(float(top.values[0] - top.values[1]), 4),
                "seconds": round(time.time() - t0, 2),
            }
        )
        if token in eos:
            break
        ids = torch.tensor([[token]])
    return steps


def teacher_decode(model: Glm53TextModel, records: list[dict]) -> torch.Tensor:
    """Each record's prompt and recorded tokens fed through the one-token decode path (the path the records came from
    and the device's decode path); the full fp32 logits at the positions that predict the recorded tokens, in record
    order.
    """
    out = []
    for r in records:
        ids = r["prompt_token_ids"] + r["generated_token_ids"]
        state = None
        prompt = len(r["prompt_token_ids"])
        with torch.no_grad():
            logits, state, _ = model(torch.tensor([r["prompt_token_ids"]]), state)
            rows = [logits[0, -1]]
            for t in range(prompt, len(ids) - 1):
                logits, state, _ = model(torch.tensor([[ids[t]]]), state)
                rows.append(logits[0, -1])
        out.append(torch.stack(rows[: len(r["generated_token_ids"])]))
        print(f"teacher-decode {r['id']}: {len(rows)} positions", flush=True)
    return torch.cat(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids", default=None, help="comma-separated prompt ids (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--teacher-decode",
        type=Path,
        default=None,
        help="instead of new records: feed the records in tools/acceptance/greedy-v1 and save the full logits here",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    checkpoint = Glm53Checkpoint(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.root)
    prompts = json.loads(PROMPTS.read_text())["prompts"]
    if args.ids:
        wanted = args.ids.split(",")
        prompts = [p for p in prompts if p["id"] in wanted]
    model = Glm53TextModel(checkpoint)
    if args.teacher_decode is not None:
        rec_dir = Path(__file__).with_name("acceptance") / "greedy-v1"
        records = [json.loads(f.read_text()) for f in sorted(rec_dir.glob("*.json"))]
        logits = teacher_decode(model, records)
        greedy = torch.tensor([t for r in records for t in r["generated_token_ids"]])
        print(
            f"argmax equals the records at {(logits.argmax(-1) == greedy).float().mean():.4%} of {len(greedy)} positions"
        )
        args.teacher_decode.parent.mkdir(parents=True, exist_ok=True)
        torch.save(logits, args.teacher_decode)
        return 0
    eos = set(model.config.eos_token_ids)
    args.out.mkdir(parents=True, exist_ok=True)
    for prompt in prompts:
        prompt_ids = render(tokenizer, prompt)
        t0 = time.time()
        steps = greedy(model, prompt_ids, args.max_new_tokens, eos)
        tokens = [s["token"] for s in steps]
        record = {
            "id": prompt["id"],
            "reference": "tt/ CPU reference = HF transformers 5.20.2 glm5_next, bf16 weights, eager experts, fp32 LM head",
            "checkpoint_revision": PINNED_REVISION,
            "messages": prompt["messages"],
            "chat_template_kwargs": prompt.get("chat_template_kwargs", {}),
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": tokens,
            "text": tokenizer.decode(tokens),
            "steps": steps,
        }
        (args.out / f"{prompt['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
        min_margin = min(s["margin"] for s in steps)
        print(
            f"{prompt['id']}: {len(prompt_ids)} prompt + {len(tokens)} tokens in {time.time() - t0:.0f}s"
            f" (prefill {steps[0]['seconds']}s), min margin {min_margin}: {record['text']!r}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
