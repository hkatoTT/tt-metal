"""AIME 2024 (HuggingFaceH4/aime_2024: AIME I and II, 30 problems, integer answers 0..999) prompts for the task-accuracy
test (tests/test_model_device_tasks.py with GLM53_TASKS_FILE set to this file's output; the GSM8K checker scores them:
the number in the last \\boxed{} after the reasoning part).

The rows come from the Hugging Face dataset viewer's rows API (data only, one page), saved as
data_dir("aime24", "raw", "rows-0.json") ($GLM53_DATA_DIR/aime24/raw/rows-0.json), e.g.

    curl -o $GLM53_DATA_DIR/aime24/raw/rows-0.json \\
        'https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/aime_2024&config=default&split=train&offset=0&length=100'

The output: data_dir("aime24", "aime24.json").  The prompt forms are gsm8k_prompts.FORMS ("max": the template's
default, a reasoning part first; "low": reasoning effort low), the user message the problem and
gsm8k_prompts.INSTRUCTION.

`--year 2025`: AIME 2025 (MathArena/aime_2025: AIME I and II, 30 problems, the same way, no solutions): its rows in
data_dir("aime25", "raw", "rows-0.json") (dataset=MathArena/aime_2025 in the URL above), the output
data_dir("aime25", "aime25.json").

    python -m models.demos.blackhole.glm53_flash.tools.aime24_prompts [--year 2024|2025] [--raw DIR] [--out JSON]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.demos.blackhole.glm53_flash.paths import data_dir
from models.demos.blackhole.glm53_flash.tools.gsm8k_prompts import FORMS, INSTRUCTION, correct

RAW = data_dir("aime24", "raw")
OUT = data_dir("aime24", "aime24.json")
SOURCES = {
    2024: (RAW, OUT, "HuggingFaceH4/aime_2024 train (30 problems), datasets-server rows API"),
    2025: (
        data_dir("aime25", "raw"),
        data_dir("aime25", "aime25.json"),
        "MathArena/aime_2025 train (30 problems), datasets-server rows API",
    ),
}


def main() -> None:
    from transformers import AutoTokenizer

    from models.demos.blackhole.glm53_flash.checkpoint import checkpoint_root

    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, choices=sorted(SOURCES), default=2024)
    p.add_argument("--raw", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    raw, out_path, source = SOURCES[args.year]
    args.raw = args.raw or raw
    args.out = args.out or out_path
    rows = []
    for f in sorted(args.raw.glob("rows-*.json")):
        rows += [r["row"] | {"idx": r["row_idx"]} for r in json.loads(f.read_text())["rows"]]
    rows = sorted(rows, key=lambda r: r["idx"])
    tok = AutoTokenizer.from_pretrained(checkpoint_root())
    items = []
    for r in rows:
        messages = [{"role": "user", "content": r["problem"] + INSTRUCTION}]
        prompts = {}
        for form, kwargs in FORMS.items():
            out = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, **kwargs
            )
            prompts[form] = [int(i) for i in out["input_ids"]]
        items.append(
            {
                "idx": r["idx"],
                "id": r.get("id", r.get("problem_idx")),
                "question": r["problem"],
                "solution": r.get("solution", ""),
                "answer": float(int(r["answer"])),
                "prompt_ids": prompts,
            }
        )
    lengths = {f: [len(x["prompt_ids"][f]) for x in items] for f in FORMS}
    doc = {
        "version": "glm53-aime%02d/v1" % (args.year % 100),
        "source": source,
        "instruction": INSTRUCTION,
        "forms": FORMS,
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False) + "\n")
    print(
        f"wrote {args.out}: {len(items)} problems; prompt tokens "
        + ", ".join(f"{f} {min(v)}..{max(v)} (mean {sum(v) / len(v):.0f})" for f, v in lengths.items())
    )
    print("max prompt 0:", tok.decode(items[0]["prompt_ids"]["max"]))
    bad = [x["idx"] for x in items if not correct("\\boxed{%d}" % int(x["answer"]), x["answer"])]
    print("checker on the answers: all correct" if not bad else f"checker fails on {bad}")


if __name__ == "__main__":
    main()
