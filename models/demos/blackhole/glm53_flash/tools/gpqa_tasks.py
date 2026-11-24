"""GPQA Diamond (Rein et al. 2023, CC BY 4.0: 198 four-choice questions) for the task-accuracy test
(tests/test_model_device_tasks.py with GLM53_TASKS_FILE set to this file's output, SCORE 0, as math500_tasks.py): the
prompts, and the accuracy of the generations on the CPU afterwards, on all 198 and on GPQA-Diamond-Clean's 189 (the
overlay adamallcock/gpqa-diamond-clean, CC BY 4.0: 9 broken items' record ids, no item text).

The questions come from the authors' own release (github.com/idavidrein/gpqa dataset.zip, the password given in its
README; GPQA's norm is no crawlable plaintext, so keep the CSV and this file's output in a private directory, never in
the repository): data_dir("gpqa") ($GLM53_DATA_DIR/gpqa, e.g. mode 700) with raw/gpqa_diamond.csv (from the zip) and
raw/exclusions.jsonl (the clean overlay's excluded record ids, one JSON object {"id": ...} a line); the output is
data_dir("gpqa", "gpqa_diamond.json") (mode 600).  The prompt is simple-evals' (the choices in an order drawn per
question from its record id, so every run the same): "Answer the following multiple choice question. The last line of
your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think
step by step before answering." then the question and the four choices; the forms gsm8k_prompts.FORMS ("max": a
reasoning part first).

`score`: the response after the reasoning part, its last "Answer: X" (simple-evals' pattern); no response (the reasoning
part did not end) or no letter: wrong.

    python -m models.demos.blackhole.glm53_flash.tools.gpqa_tasks prompts [--raw DIR] [--out JSON]
    python -m models.demos.blackhole.glm53_flash.tools.gpqa_tasks score RESULTS.json ... [--out JSON]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

from models.demos.blackhole.glm53_flash.paths import data_dir
from models.demos.blackhole.glm53_flash.tools.gsm8k_prompts import FORMS

RAW = data_dir("gpqa", "raw")
OUT = data_dir("gpqa", "gpqa_diamond.json")
TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
    "{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)
ANSWER = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?\(?([A-D])\)?\$?")


def prompts(args) -> None:
    from transformers import AutoTokenizer

    from models.demos.blackhole.glm53_flash.checkpoint import checkpoint_root

    with open(args.raw / "gpqa_diamond.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    excluded = {json.loads(line)["id"] for line in (args.raw / "exclusions.jsonl").read_text().splitlines() if line}
    tok = AutoTokenizer.from_pretrained(checkpoint_root())
    items = []
    for i, r in enumerate(rows):
        choices = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        choices = [c.strip() for c in choices]
        seed = int(hashlib.sha256(r["Record ID"].encode()).hexdigest()[:16], 16)
        order = random.Random(seed).sample(range(4), 4)
        letters = dict(zip("ABCD", (choices[j] for j in order)))
        content = TEMPLATE.format(question=r["Question"].strip(), **letters)
        ids = {}
        for form, kwargs in FORMS.items():
            out = tok.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True, return_dict=True,
                **kwargs,
            )  # fmt: skip
            ids[form] = [int(x) for x in out["input_ids"]]
        items.append(
            {
                "idx": i,
                "record_id": r["Record ID"],
                "domain": r.get("High-level domain", ""),
                "letter": "ABCD"[order.index(0)],
                "clean": r["Record ID"] not in excluded,
                "answer": float("nan"),  # (the test's numeric checker scores none)
                "prompt_ids": ids,
            }
        )
    lengths = {f: [len(x["prompt_ids"][f]) for x in items] for f in FORMS}
    doc = {"version": "glm53-gpqa-diamond/v1", "source": "idavidrein/gpqa dataset.zip (CC BY 4.0)", "items": items}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc) + "\n")
    args.out.chmod(0o600)
    print(
        f"wrote {args.out}: {len(items)} questions ({sum(x['clean'] for x in items)} in the clean set); letters "
        + ", ".join(f"{c} {sum(x['letter'] == c for x in items)}" for c in "ABCD")
        + "; prompt tokens "
        + ", ".join(f"{f} {min(v)}..{max(v)} (mean {sum(v) / len(v):.0f})" for f, v in lengths.items())
    )


def pick(text: str) -> str | None:
    if "</think>" not in text:
        return None
    found = ANSWER.findall(text.split("</think>", 1)[1])
    return found[-1].upper() if found else None


def score(args) -> None:
    from tokenizers import Tokenizer

    from models.demos.blackhole.glm53_flash.checkpoint import checkpoint_root

    tok = Tokenizer.from_file(str(Path(checkpoint_root()) / "tokenizer.json"))
    items = {x["idx"]: x for x in json.loads(OUT.read_text())["items"]}
    out = {}
    for path in args.results:
        doc = json.loads(Path(path).read_text())
        ids = doc.get("problem_ids")
        for r in doc["results"]:
            unfinished = r.get("unfinished") or {}
            got = {}
            for k, gen in list(r["generations"].items()) + [(k, None) for k in unfinished]:
                idx = ids[int(k)] if ids else int(k)
                if idx not in items:
                    continue
                letter = pick(tok.decode(gen, skip_special_tokens=True)) if gen is not None else None
                got[idx] = {
                    "letter": letter,
                    "correct": letter == items[idx]["letter"],
                    "tokens": len(gen) if gen is not None else len(unfinished[k]),
                    "ended": gen is not None and k not in unfinished,
                }
            out[f"{Path(path).name}:{r['mode']}"] = got
            for name, sel in (("198", sorted(got)), ("clean 189", [i for i in sorted(got) if items[i]["clean"]])):
                rows = [got[i] for i in sel]
                print(
                    f"{Path(path).name} {r['mode']} {name}: {sum(g['correct'] for g in rows)} / {len(rows)} = "
                    f"{sum(g['correct'] for g in rows) / max(1, len(rows)):.3f} ({sum(g['letter'] is None for g in rows)} "
                    "without a letter)",
                    flush=True,
                )
    if args.out:
        args.out.write_text(json.dumps(out) + "\n")
        print(f"wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prompts")
    a.add_argument("--raw", type=Path, default=RAW)
    a.add_argument("--out", type=Path, default=OUT)
    b = sub.add_parser("score")
    b.add_argument("results", nargs="+")
    b.add_argument("--out", type=Path)
    args = p.parse_args()
    {"prompts": prompts, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
