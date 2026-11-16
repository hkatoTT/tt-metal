"""HumanEval (openai/openai_humaneval, MIT) for the task-accuracy test (tests/test_model_device_tasks.py with
GLM53_TASKS_SET=humaneval): the prompts, and pass@1 of the generations on the CPU.

The 164 problems come from the Hugging Face dataset viewer's rows API (data only, two pages of up to 100 rows), saved
as data_dir("humaneval", "raw") / rows-<offset>.json ($GLM53_DATA_DIR/humaneval/raw/rows-0.json and rows-100.json),
e.g.

    curl -o $GLM53_DATA_DIR/humaneval/raw/rows-0.json \\
        'https://datasets-server.huggingface.co/rows?dataset=openai/openai_humaneval&config=openai_humaneval&split=test&offset=0&length=100'

The output, data_dir("humaneval", "humaneval164.json"), is the test's default file for GLM53_TASKS_SET=humaneval.
The prompt: the problem's function stub in a ```python block and an instruction to write the whole function, rendered
with reasoning_effort "low" (as GSM8K's).

    python -m models.demos.blackhole.glm53_flash.tools.humaneval_tasks prompts [--raw DIR] [--out JSON]
    python -m models.demos.blackhole.glm53_flash.tools.humaneval_tasks score RESULTS.json ... [--out JSON]
    python -m models.demos.blackhole.glm53_flash.tools.humaneval_tasks selftest

`score` (the CPU only, outside any device run): each form's generations decoded, the code taken from the answer (the
```python block that defines the entry point, else the last block, else the text after the reasoning part) and put
after the whole stub, so that the stub's imports and helper functions are there (a body without the `def`: under the
stub), and run with the problem's test in its own process: `python3 -I` in a fresh temporary directory (removed
afterwards), a minimal environment, 10 s wall clock and CPU time, 2 GB of address space, 10 MB file size
(`resource.setrlimit`), its process group killed on a timeout.  The generated code is the model's: it
runs unreviewed, so only here, one problem at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from models.demos.blackhole.glm53_flash.paths import data_dir

RAW = data_dir("humaneval", "raw")
OUT = data_dir("humaneval", "humaneval164.json")
INSTRUCTION = (
    "Complete the following Python function. Write the whole function, with any imports it needs, in one ```python "
    "code block.\n\n```python\n{prompt}```"
)
TIMEOUT = 10
MEMORY = 2 << 30


def prompts(args) -> None:
    from transformers import AutoTokenizer

    from models.demos.blackhole.glm53_flash.checkpoint import checkpoint_root

    rows = []
    for f in sorted(args.raw.glob("rows-*.json"), key=lambda f: int(f.stem.split("-")[1])):
        rows += [r["row"] | {"idx": r["row_idx"]} for r in json.loads(f.read_text())["rows"]]
    tok = AutoTokenizer.from_pretrained(checkpoint_root())
    items = []
    for r in sorted(rows, key=lambda r: r["idx"]):
        messages = [{"role": "user", "content": INSTRUCTION.format(prompt=r["prompt"])}]
        ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, reasoning_effort="low"
        )["input_ids"]
        items.append(
            {
                "idx": r["idx"],
                "task_id": r["task_id"],
                "prompt": r["prompt"],
                "test": r["test"],
                "canonical_solution": r["canonical_solution"],  # (the harness's self-test only)
                "entry_point": r["entry_point"],
                "answer": None,
                "prompt_ids": {"low": [int(i) for i in ids]},
            }
        )
    lengths = [len(x["prompt_ids"]["low"]) for x in items]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "version": "glm53-humaneval/v1",
                "source": "openai/openai_humaneval test (MIT), datasets-server rows API",
                "instruction": INSTRUCTION,
                "items": items,
            }
        )
        + "\n"
    )
    print(f"wrote {args.out}: {len(items)} problems, prompt tokens {min(lengths)}..{max(lengths)}")
    print(tok.decode(items[0]["prompt_ids"]["low"]))


def extract_code(text: str, item: dict) -> str:
    """The program's function from the model's answer (see the module's docstring)."""
    tail = text.split("</think>")[-1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", tail, flags=re.S)
    if not blocks:  # (an unclosed block: the text after its fence)
        m = re.search(r"```(?:python|py)?\s*\n(.*)", tail, flags=re.S)
        blocks = [m.group(1)] if m else [tail]
    entry = f"def {item['entry_point']}"
    code = next((b for b in blocks if entry in b), blocks[-1])
    if entry not in code:  # a body only: under the stub
        return item["prompt"] + code
    # the whole function after the whole stub (its imports and helpers, such as problem 38's encode_cyclic, in case the
    # answer left them out; the stub's own `def` with only its docstring is valid and the answer's replaces it)
    return item["prompt"] + "\n\n" + code


def _limits() -> None:
    os.setsid()
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT, TIMEOUT))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))


def run_one(program: str) -> tuple[bool, str]:
    """The program in its own process, in a temporary directory removed afterwards -> (passed, why not)."""
    work = tempfile.mkdtemp(prefix="humaneval-")
    try:
        path = Path(work) / "program.py"
        path.write_text(program)
        env = {"PATH": "/usr/bin:/bin", "HOME": work, "TMPDIR": work, "PYTHONHASHSEED": "0", "LANG": "C.UTF-8"}
        proc = subprocess.Popen(
            [sys.executable if os.environ.get("HUMANEVAL_PYTHON") is None else os.environ["HUMANEVAL_PYTHON"],
             "-I", "-B", str(path)],
            cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            preexec_fn=_limits,
        )  # fmt: skip
        try:
            _, err = proc.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return False, "timeout"
        if proc.returncode == 0:
            return True, ""
        last = err.decode(errors="replace").strip().splitlines()
        return False, (last[-1] if last else f"exit {proc.returncode}")[:200]
    finally:
        shutil.rmtree(work, ignore_errors=True)


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
            got = out.setdefault(r["mode"], {})
            for k, gen in r["generations"].items():
                idx = ids[int(k)] if ids else int(k)
                if idx not in items:
                    continue
                text = tok.decode(gen, skip_special_tokens=True)
                program = extract_code(text, items[idx]) + "\n\n" + items[idx]["test"]
                program += f"\n\ncheck({items[idx]['entry_point']})\n"
                passed, why = run_one(program)
                got[str(idx)] = {"passed": passed, "why": why, "tokens": len(gen)}
            n = len(got)
            print(f"{Path(path).name} {r['mode']}: pass@1 {sum(v['passed'] for v in got.values())} / {n}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(out) + "\n")
        print(f"wrote {args.out}")


def selftest(args) -> None:
    """The harness on the reference solutions (all must pass), on stubs returning None (most must fail), and on the
    answer extraction of a fenced whole function."""
    items = json.loads(OUT.read_text())["items"]
    ok = bad = extracted = 0
    for x in items:
        tail = x["test"] + f"\n\ncheck({x['entry_point']})\n"
        ok += run_one(x["prompt"] + x["canonical_solution"] + "\n\n" + tail)[0]
        bad += run_one(x["prompt"] + "    return None\n\n" + tail)[0]
        answer = "</think>Here it is.\n```python\n" + x["prompt"] + x["canonical_solution"] + "```\nDone."
        extracted += run_one(extract_code(answer, x) + "\n\n" + tail)[0]
    print(
        f"reference solutions pass {ok} / {len(items)}; stubs returning None pass {bad}; fenced answers pass "
        f"{extracted} / {len(items)}"
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
