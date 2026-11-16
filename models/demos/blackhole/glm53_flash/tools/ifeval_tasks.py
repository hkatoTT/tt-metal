"""IFEval (google/IFEval, Apache 2.0: 541 prompts with verifiable instructions; also scored on the first 100) for the
task-accuracy test (tests/test_model_device_tasks.py with GLM53_TASKS_FILE set to this file's output, as
math500_tasks.py; SCORE 0): the prompts, and the scores of the generations on the CPU afterwards.

The rows come from the Hugging Face dataset viewer's rows API (data only, six pages of up to 100 rows), saved as
data_dir("ifeval", "raw") / rows-<offset>.json ($GLM53_DATA_DIR/ifeval/raw/rows-0.json .. rows-500.json), e.g.

    curl -o $GLM53_DATA_DIR/ifeval/raw/rows-0.json \\
        'https://datasets-server.huggingface.co/rows?dataset=google/IFEval&config=default&split=train&offset=0&length=100'

The output: data_dir("ifeval", "ifeval.json").  The prompt forms are gsm8k_prompts.FORMS ("max": the template's default,
a reasoning part first; "low": reasoning effort low), the user message the prompt alone (no answer format added).

`score`: each form's generations decoded, the response the text after the reasoning part (none when the reasoning part
did not end: every instruction fails), checked by the official checker (google-research's instruction_following_eval,
Apache 2.0: its evaluation_lib, strict and loose) from IFEVAL_DIR (default data_dir("ifeval", "checker"): the parent
of its instruction_following_eval package directory, e.g. from github.com/google-research/google-research) in the venv
IFEval's checker needs (absl-py, langdetect, nltk with punkt, immutabledict): prompt-level and instruction-level
accuracy, strict and loose, on all prompts and on the first 100.

    python -m models.demos.blackhole.glm53_flash.tools.ifeval_tasks prompts [--raw DIR] [--out JSON]
    python -m models.demos.blackhole.glm53_flash.tools.ifeval_tasks score RESULTS.json ... [--out JSON]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

from models.demos.blackhole.glm53_flash.paths import checkpoint_dir, data_dir
from models.demos.blackhole.glm53_flash.tools.gsm8k_prompts import FORMS

RAW = data_dir("ifeval", "raw")
OUT = data_dir("ifeval", "ifeval.json")
IFEVAL_DIR = Path(os.environ.get("IFEVAL_DIR") or data_dir("ifeval", "checker"))


def _rows(raw: Path) -> list[dict]:
    """The dataset viewer's rows in order, each instruction's kwargs without the parameters it does not take (the
    viewer fills every row's dicts with all the parameters, None where unused)."""
    rows = []
    for f in sorted(raw.glob("rows-*.json")):
        rows += [r["row"] | {"idx": r["row_idx"]} for r in json.loads(f.read_text())["rows"]]
    rows = sorted(rows, key=lambda r: r["idx"])
    for r in rows:
        for k in ("instruction_id_list", "kwargs"):
            if isinstance(r[k], str):
                r[k] = ast.literal_eval(r[k])
        r["kwargs"] = [{a: v for a, v in kw.items() if v is not None} for kw in r["kwargs"]]
    return rows


def prompts(args) -> None:
    from transformers import AutoTokenizer

    from models.demos.blackhole.glm53_flash.checkpoint import checkpoint_root

    rows = _rows(args.raw)
    tok = AutoTokenizer.from_pretrained(checkpoint_root())
    items = []
    for r in rows:
        messages = [{"role": "user", "content": r["prompt"]}]
        ids = {}
        for form, kwargs in FORMS.items():
            out = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, **kwargs
            )
            ids[form] = [int(i) for i in out["input_ids"]]
        items.append(
            {
                "idx": r["idx"],
                "key": int(r["key"]),
                "question": r["prompt"],
                "instruction_id_list": r["instruction_id_list"],
                "kwargs": r["kwargs"],
                "answer": float("nan"),  # (no numeric answer: the test's checker scores none)
                "prompt_ids": ids,
            }
        )
    lengths = {f: [len(x["prompt_ids"][f]) for x in items] for f in FORMS}
    doc = {
        "version": "glm53-ifeval/v1",
        "source": "google/IFEval train (541 prompts), datasets-server rows API",
        "forms": FORMS,
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False) + "\n")
    print(
        f"wrote {args.out}: {len(items)} prompts, {sum(len(x['instruction_id_list']) for x in items)} instructions; "
        "prompt tokens " + ", ".join(f"{f} {min(v)}..{max(v)} (mean {sum(v) / len(v):.0f})" for f, v in lengths.items())
    )


def score(args) -> None:
    """(Runs in the checker's venv: the tokenizer read from $GLM53_CHECKPOINT, no torch.)"""
    from tokenizers import Tokenizer

    sys.path.insert(0, str(IFEVAL_DIR))
    from instruction_following_eval import evaluation_lib as ev

    tok = Tokenizer.from_file(str(checkpoint_dir() / "tokenizer.json"))
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
                text = tok.decode(gen, skip_special_tokens=True) if gen is not None else ""
                ended = gen is not None and k not in unfinished
                response = text.split("</think>", 1)[1].strip() if ended and "</think>" in text else ""
                x = items[idx]
                inp = ev.InputExample(
                    key=x["key"], instruction_id_list=x["instruction_id_list"], prompt=x["question"], kwargs=x["kwargs"]
                )
                strict = ev.test_instruction_following_strict(inp, {x["question"]: response})
                loose = ev.test_instruction_following_loose(inp, {x["question"]: response})
                got[idx] = {
                    "strict": strict.follow_instruction_list,
                    "loose": loose.follow_instruction_list,
                    "ended": bool(response),
                    "tokens": len(gen) if gen is not None else len(unfinished[k]),
                }
            out[f"{Path(path).name}:{r['mode']}"] = got
            for name, sel in (("all", sorted(got)), ("first 100", [i for i in sorted(got) if i < 100])):
                rows = [got[i] for i in sel]
                if not rows:
                    continue
                n_ins = sum(len(g["strict"]) for g in rows)
                print(
                    f"{Path(path).name} {r['mode']} {name} ({len(rows)} prompts, {sum(g['ended'] for g in rows)} with "
                    f"a response): prompt strict {sum(all(g['strict']) for g in rows) / len(rows):.3f}, loose "
                    f"{sum(all(g['loose']) for g in rows) / len(rows):.3f}; instruction strict "
                    f"{sum(sum(g['strict']) for g in rows) / n_ins:.3f}, loose {sum(sum(g['loose']) for g in rows) / n_ins:.3f}",
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
