"""MBPP (google-research-datasets/mbpp, CC-BY-4.0: the full config's test split, 500 problems) for the task-accuracy test
(tests/test_model_device_tasks.py with GLM53_TASKS_FILE set to this file's output, form "low", SCORE 0, as HumanEval's
runs): the prompts, and pass@1 of the generations on the CPU afterwards.

The rows come from the Hugging Face dataset viewer's rows API (data only, five pages of up to 100 rows), saved as
data_dir("mbpp", "raw") / rows-<offset>.json ($GLM53_DATA_DIR/mbpp/raw/rows-0.json .. rows-400.json), e.g.

    curl -o $GLM53_DATA_DIR/mbpp/raw/rows-0.json \\
        'https://datasets-server.huggingface.co/rows?dataset=google-research-datasets/mbpp&config=full&split=test&offset=0&length=100'

The output: data_dir("mbpp", "mbpp500.json").  The prompt (the usual MBPP form): the problem's text, its three tests (so
that the function's name and arguments are known) and an instruction to write the function in one ```python block,
rendered with reasoning_effort "low" (as HumanEval's).

`score` (the CPU only, outside any device run): each form's generations decoded, the code the ```python block after the
reasoning part (the one that defines the tested function, else the last, else the text), then the problem's setup code
and its three asserts, run by humaneval_tasks.run_one (its own process, `python3 -I` in a fresh temporary directory, 10 s,
2 GB).  The generated code is the model's: it runs unreviewed, so only here, one problem at a time.

    python -m models.demos.blackhole.glm53_flash.tools.mbpp_tasks prompts [--raw DIR] [--out JSON]
    python -m models.demos.blackhole.glm53_flash.tools.mbpp_tasks score RESULTS.json ... [--out JSON]
    python -m models.demos.blackhole.glm53_flash.tools.mbpp_tasks selftest
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from models.demos.blackhole.glm53_flash.paths import data_dir
from models.demos.blackhole.glm53_flash.tools.humaneval_tasks import run_one

RAW = data_dir("mbpp", "raw")
OUT = data_dir("mbpp", "mbpp500.json")
INSTRUCTION = "{text}\nYour code should pass these tests:\n\n{tests}\n\nWrite the function in one ```python code block."


def _tested_name(tests: list[str]) -> str | None:
    m = re.search(r"assert\s+(?:not\s+)?\(?\s*(?:set\()?\s*([A-Za-z_]\w*)\s*\(", tests[0]) if tests else None
    return m.group(1) if m else None


def prompts(args) -> None:
    from transformers import AutoTokenizer

    from models.demos.blackhole.glm53_flash.checkpoint import checkpoint_root

    rows = []
    for f in sorted(args.raw.glob("rows-*.json"), key=lambda f: int(f.stem.split("-")[1])):
        rows += [r["row"] | {"idx": r["row_idx"]} for r in json.loads(f.read_text())["rows"]]
    tok = AutoTokenizer.from_pretrained(checkpoint_root())
    items = []
    for r in sorted(rows, key=lambda r: r["idx"]):
        tests = list(r["test_list"])
        messages = [{"role": "user", "content": INSTRUCTION.format(text=r["text"], tests="\n".join(tests))}]
        ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, reasoning_effort="low"
        )["input_ids"]
        items.append(
            {
                "idx": r["idx"],
                "task_id": r["task_id"],
                "text": r["text"],
                "tests": tests,
                "setup": r.get("test_setup_code") or "",
                "code": r["code"],  # (the reference: the harness's self-test only)
                "name": _tested_name(tests),
                "answer": float("nan"),
                "prompt_ids": {"low": [int(i) for i in ids]},
            }
        )
    lengths = [len(x["prompt_ids"]["low"]) for x in items]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": "glm53-mbpp/v1",
        "source": "google-research-datasets/mbpp full test (CC-BY-4.0, 500 problems), datasets-server rows API",
        "instruction": INSTRUCTION,
        "items": items,
    }
    args.out.write_text(json.dumps(doc) + "\n")
    print(f"wrote {args.out}: {len(items)} problems, prompt tokens {min(lengths)}..{max(lengths)}")
    print(tok.decode(items[0]["prompt_ids"]["low"]))


def extract_code(text: str, item: dict) -> str:
    tail = text.split("</think>")[-1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", tail, flags=re.S)
    if not blocks:  # (an unclosed block: the text after its fence)
        m = re.search(r"```(?:python|py)?\s*\n(.*)", tail, flags=re.S)
        blocks = [m.group(1)] if m else [tail]
    name = item.get("name")
    return next((b for b in blocks if name and f"def {name}" in b), blocks[-1])


def program(code: str, item: dict) -> str:
    return code + "\n\n" + item["setup"] + "\n\n" + "\n".join(item["tests"]) + "\n"


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
            got = out.setdefault(f"{Path(path).name}:{r['mode']}", {})
            unfinished = r.get("unfinished") or {}
            for k, gen in r["generations"].items():
                idx = ids[int(k)] if ids else int(k)
                if idx not in items:
                    continue
                text = tok.decode(gen, skip_special_tokens=True)
                passed, why = run_one(program(extract_code(text, items[idx]), items[idx]))
                got[str(idx)] = {"passed": passed, "why": why, "tokens": len(gen), "ended": k not in unfinished}
            for k in unfinished:  # (cut by the time budget: failed)
                idx = ids[int(k)] if ids else int(k)
                got.setdefault(
                    str(idx), {"passed": False, "why": "unfinished", "tokens": len(unfinished[k]), "ended": False}
                )
            print(
                f"{Path(path).name} {r['mode']}: pass@1 {sum(v['passed'] for v in got.values())} / {len(got)} "
                f"({sum(not v['ended'] for v in got.values())} unfinished)",
                flush=True,
            )
    if args.out:
        args.out.write_text(json.dumps(out) + "\n")
        print(f"wrote {args.out}")


def selftest(args) -> None:
    """The harness on the reference solutions (nearly all must pass), on stubs (most must fail), and on the extraction of
    a fenced answer."""
    items = json.loads(OUT.read_text())["items"]
    ok = bad = extracted = 0
    for x in items:
        ok += run_one(program(x["code"], x))[0]
        bad += run_one(program(f"def {x['name']}(*args, **kwargs):\n    return None\n", x))[0]
        answer = "</think>Here it is.\n```python\n" + x["code"] + "\n```\nDone."
        extracted += run_one(program(extract_code(answer, x), x))[0]
    print(
        f"reference solutions pass {ok} / {len(items)}; stubs returning None pass {bad}; fenced answers pass "
        f"{extracted} / {len(items)}; no tested name found: {sum(x['name'] is None for x in items)}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    a = sub.add_parser("prompts")
    a.add_argument("--raw", type=Path, default=RAW)
    a.add_argument("--out", type=Path, default=OUT)
    b = sub.add_parser("score")
    b.add_argument("results", nargs="+")
    b.add_argument("--out", type=Path)
    args = p.parse_args()
    {"prompts": prompts, "score": score, "selftest": selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
