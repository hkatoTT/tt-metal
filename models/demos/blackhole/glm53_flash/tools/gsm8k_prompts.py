"""GSM8K (openai/gsm8k, MIT) prompts for the task-accuracy test (tests/test_model_device_tasks.py): the first problems
of the test split, each rendered with the checkpoint's chat template, and the answer checker.

The rows come from the Hugging Face dataset viewer's rows API (data only, pages of up to 100 rows), saved as
data_dir("gsm8k", "raw") / rows-<offset>.json ($GLM53_DATA_DIR/gsm8k/raw/rows-0.json, rows-100.json, ...), e.g.

    curl -o $GLM53_DATA_DIR/gsm8k/raw/rows-0.json \\
        'https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=main&split=test&offset=0&length=100'

The output, data_dir("gsm8k", "gsm8k-test200.json"), is the test's default GLM53_TASKS_FILE.  Each problem keeps its
question, the reference solution and its final number (after "####").  The prompt forms (`FORMS`): the user message is
the question and an instruction to put the final answer in \\boxed{}; "low" renders it with reasoning_effort "low" (the
template's "Reasoning Effort: Low": the model closes its <think> at once and answers, ~100-300 tokens, the greedy
record math-low), "max" with the template's default (a reasoning part first; longer).

    python -m models.demos.blackhole.glm53_flash.tools.gsm8k_prompts [--raw DIR] [--out JSON] [--n 200]

The checker (`extract_answer`, `correct`): the number in the last \\boxed{..} of the text after the reasoning part, else
the text's last number; commas, a leading $ and a trailing .0 dropped; compared as numbers.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from models.demos.blackhole.glm53_flash.paths import data_dir

RAW = data_dir("gsm8k", "raw")
OUT = data_dir("gsm8k", "gsm8k-test200.json")
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
FORMS = {"low": {"reasoning_effort": "low"}, "max": {}}
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _number(text: str) -> float | None:
    m = _NUMBER.findall(text)
    if not m:
        return None
    s = m[-1].replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    """The model's final number: inside the last \\boxed{..} after the reasoning part (</think>), else its last number."""
    tail = text.split("</think>")[-1]
    boxed = list(re.finditer(r"\\boxed\{", tail))
    if boxed:
        start = boxed[-1].end()
        depth, i = 1, start
        while i < len(tail) and depth:
            depth += {"{": 1, "}": -1}.get(tail[i], 0)
            i += 1
        got = _number(tail[start : i - 1] if depth == 0 else tail[start:])
        if got is not None:
            return got
    return _number(tail)


def reference_answer(solution: str) -> float:
    return float(solution.split("####")[-1].strip().replace(",", ""))


def correct(text: str, answer: float) -> bool:
    got = extract_answer(text)
    return got is not None and abs(got - answer) < 1e-6 * max(1.0, abs(answer))


def main() -> None:
    from transformers import AutoTokenizer

    from models.demos.blackhole.glm53_flash.checkpoint import checkpoint_root

    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    rows = []
    for f in sorted(args.raw.glob("rows-*.json"), key=lambda f: int(f.stem.split("-")[1])):
        rows += [r["row"] | {"idx": r["row_idx"]} for r in json.loads(f.read_text())["rows"]]
    rows = sorted(rows, key=lambda r: r["idx"])[: args.n]
    tok = AutoTokenizer.from_pretrained(checkpoint_root())
    items = []
    for r in rows:
        messages = [{"role": "user", "content": r["question"] + INSTRUCTION}]
        prompts = {}
        for form, kwargs in FORMS.items():
            out = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, **kwargs
            )
            prompts[form] = [int(i) for i in out["input_ids"]]
        items.append(
            {
                "idx": r["idx"],
                "question": r["question"],
                "solution": r["answer"],
                "answer": reference_answer(r["answer"]),
                "prompt_ids": prompts,
            }
        )
    lengths = {f: [len(x["prompt_ids"][f]) for x in items] for f in FORMS}
    doc = {
        "version": "glm53-gsm8k-test/v1",
        "source": "openai/gsm8k main test (MIT), rows 0..%d, datasets-server rows API" % (len(items) - 1),
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
    print("low prompt 0:", tok.decode(items[0]["prompt_ids"]["low"]))
    print("max prompt 0:", tok.decode(items[0]["prompt_ids"]["max"]))
    bad = [
        x["idx"]
        for x in items
        if not correct(x["solution"].split("####")[0] + "\\boxed{%s}" % x["answer"], x["answer"])
    ]
    print("checker on the reference solutions: all correct" if not bad else f"checker fails on {bad}")


if __name__ == "__main__":
    main()
