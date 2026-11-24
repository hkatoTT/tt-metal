"""MATH-500 (HuggingFaceH4/MATH-500, MIT; the first 100 problems) for the task-accuracy test
(tests/test_model_device_tasks.py with GLM53_TASKS_FILE set to this file's output, as aime24_prompts.py): the prompts,
and the accuracy of the generations on the CPU afterwards (the answers are LaTeX: the test's numeric checker scores only
the numeric ones).

The rows come from the Hugging Face dataset viewer's rows API (data only, one page of 100 rows), saved as
data_dir("math500", "raw", "rows-0.json") ($GLM53_DATA_DIR/math500/raw/rows-0.json), e.g.

    curl -o $GLM53_DATA_DIR/math500/raw/rows-0.json \\
        'https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500&config=default&split=test&offset=0&length=100'

The output: data_dir("math500", "math500.json").  The prompt forms are gsm8k_prompts.FORMS ("max": the template's
default, a reasoning part first; "low": reasoning effort low), the user message the problem and
gsm8k_prompts.INSTRUCTION.

`score`: each form's generations decoded, the last \\boxed{..} after the reasoning part taken and compared with the
answer after the MATH dataset's normalization (Hendrycks et al.'s `strip_string`: \\left / \\right, \\dfrac, units in
\\text{..}, degrees, spaces, "x =", \\frac12, 0.5 / a/b forms), then as numbers when both read as one.

    python -m models.demos.blackhole.glm53_flash.tools.math500_tasks prompts [--raw DIR] [--out JSON]
    python -m models.demos.blackhole.glm53_flash.tools.math500_tasks score RESULTS.json ... [--out JSON]
    python -m models.demos.blackhole.glm53_flash.tools.math500_tasks selftest
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from models.demos.blackhole.glm53_flash.paths import data_dir
from models.demos.blackhole.glm53_flash.tools.gsm8k_prompts import FORMS, INSTRUCTION

RAW = data_dir("math500", "raw")
OUT = data_dir("math500", "math500.json")


def last_boxed(text: str) -> str | None:
    """The content of the last \\boxed{..} (braces matched) after the reasoning part (</think>), else None."""
    tail = text.split("</think>")[-1]
    starts = list(re.finditer(r"\\boxed\s*\{", tail))
    if not starts:
        return None
    i = start = starts[-1].end()
    depth = 1
    while i < len(tail) and depth:
        depth += {"{": 1, "}": -1}.get(tail[i], 0)
        i += 1
    return tail[start : i - 1] if depth == 0 else tail[start:]


def _fix_fracs(s: str) -> str:
    """\\frac12 -> \\frac{1}{2}, \\frac1{2} -> \\frac{1}{2} (as `strip_string`)."""
    parts = s.split("\\frac")
    out = parts[0]
    for p in parts[1:]:
        out += "\\frac"
        if p.startswith("{"):
            out += p
        elif len(p) < 2:
            return s
        elif p[1] == "{":
            out += "{" + p[0] + "}" + p[1:]
        else:
            out += "{" + p[0] + "}{" + p[1] + "}" + p[2:]
    return out


def _fix_a_slash_b(s: str) -> str:
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    return "\\frac{%s}{%s}" % (m.group(1), m.group(2)) if m else s


def _fix_sqrt(s: str) -> str:
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)


def normalize(s: str) -> str:
    """The MATH dataset's `strip_string` (Hendrycks et al., MIT), with \\text{..} unwrapped."""
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\{\s*([^{}]*?)\s*\}", r"\1", s)
    s = re.sub(r"\s*(?:units?|cm|meters?|inches|feet|degrees|dollars|cents)\s*$", "", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if s.count("=") == 1 and len(s.split("=")[0].strip()) <= 2:  # "x = 5" -> "5"
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace("\\ ", "").replace("\\,", "").replace("\\;", "").replace("\\quad", "").replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    s = s.rstrip(".")
    return s


def _number(s: str) -> float | None:
    """A normalized answer as one number: an integer / decimal (thousands commas allowed), or \\frac{a}{b}."""
    t = s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s) else s
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}", t)
    if m:
        v = float(m.group(2)) / float(m.group(3))
        return -v if m.group(1) else v
    try:
        return float(t)
    except ValueError:
        return None


def _same_one(a: str, b: str) -> bool:
    if a == b or a.lower() == b.lower():  # (\text{East} for \text{east})
        return True
    x, y = _number(a), _number(b)
    return x is not None and y is not None and math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)


def _items(s: str) -> list[str] | None:
    """A normalized answer as an unordered list of values (the solutions of an equation: "1,-2", "a\\pm b" as a-b and
    a+b), or None for one value or an ordered tuple / interval (in brackets)."""
    if s[:1] in "([{" and s[-1:] in ")]}":
        return None
    m = re.fullmatch(r"(.*)\\pm(.*)", s)
    if m and "," not in s:
        return [m.group(1) + "-" + m.group(2), m.group(1) + "+" + m.group(2)]
    depth, parts, cur = 0, [], ""
    for ch in s:
        depth += {"{": 1, "(": 1, "[": 1, "}": -1, ")": -1, "]": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts if len(parts) > 1 else None


def equal(got: str | None, answer: str) -> bool:
    if got is None:
        return False
    a, b = normalize(got), normalize(answer)
    if _same_one(a, b):
        return True
    ia, ib = _items(a), _items(b)
    if ia is None or ib is None or len(ia) != len(ib):
        return False
    left = list(ib)
    for x in ia:  # (each value matched once, in any order)
        hit = next((y for y in left if _same_one(x, y)), None)
        if hit is None:
            return False
        left.remove(hit)
    return True


def prompts(args) -> None:
    from transformers import AutoTokenizer

    from models.demos.blackhole.glm53_flash.checkpoint import checkpoint_root

    rows = []
    for f in sorted(args.raw.glob("rows-*.json")):
        rows += [r["row"] | {"idx": r["row_idx"]} for r in json.loads(f.read_text())["rows"]]
    rows = sorted(rows, key=lambda r: r["idx"])
    tok = AutoTokenizer.from_pretrained(checkpoint_root())
    items = []
    for r in rows:
        messages = [{"role": "user", "content": r["problem"] + INSTRUCTION}]
        ids = {}
        for form, kwargs in FORMS.items():
            out = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, **kwargs
            )
            ids[form] = [int(i) for i in out["input_ids"]]
        number = _number(normalize(r["answer"]))
        items.append(
            {
                "idx": r["idx"],
                "id": r["unique_id"],
                "subject": r["subject"],
                "level": r["level"],
                "question": r["problem"],
                "solution": r["solution"],
                "answer_text": r["answer"],
                # (the test's numeric checker: the numeric answers only; NaN never matches)
                "answer": number if number is not None else float("nan"),
                "prompt_ids": ids,
            }
        )
    lengths = {f: [len(x["prompt_ids"][f]) for x in items] for f in FORMS}
    doc = {
        "version": "glm53-math500/v1",
        "source": "HuggingFaceH4/MATH-500 test, rows 0..%d, datasets-server rows API" % (len(items) - 1),
        "instruction": INSTRUCTION,
        "forms": FORMS,
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False) + "\n")
    print(
        f"wrote {args.out}: {len(items)} problems ({sum(not math.isnan(x['answer']) for x in items)} numeric); prompt "
        "tokens " + ", ".join(f"{f} {min(v)}..{max(v)} (mean {sum(v) / len(v):.0f})" for f, v in lengths.items())
    )


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
            unfinished = set(r.get("unfinished") or {})
            for k, gen in r["generations"].items():
                idx = ids[int(k)] if ids else int(k)
                if idx not in items:
                    continue
                text = tok.decode(gen, skip_special_tokens=True)
                boxed = last_boxed(text)
                got[str(idx)] = {
                    "correct": equal(boxed, items[idx]["answer_text"]),
                    "boxed": boxed,
                    "answer": items[idx]["answer_text"],
                    "tokens": len(gen),
                    "ended": k not in unfinished,
                }
            n = len(got)
            ok = sum(v["correct"] for v in got.values())
            ended = sum(v["ended"] for v in got.values())
            print(f"{Path(path).name} {r['mode']}: {ok} / {n} correct ({ended} ended)", flush=True)
    if args.out:
        args.out.write_text(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


def selftest(args) -> None:
    """The checker on each reference solution's own last \\boxed{} (must all match), and on a few known forms."""
    items = json.loads(OUT.read_text())["items"]
    bad = [x["idx"] for x in items if not equal(last_boxed(x["solution"]), x["answer_text"])]
    cases = [
        ("\\dfrac{3}{56}", "\\frac{3}{56}", True),
        ("3/56", "\\frac{3}{56}", True),
        ("90", "90^\\circ", True),
        ("x=5", "5", True),
        ("\\text{Evelyn}", "\\text{Evelyn}", True),
        ("Evelyn", "\\text{Evelyn}", True),
        ("2,220", "2220", True),
        ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})", True),
        ("6-5i", "6 - 5i", True),
        ("0.5", "\\frac12", True),
        ("4", "5", False),
        ("\\frac{3}{5}", "\\frac{5}{3}", False),
    ]
    wrong = [(a, b) for a, b, want in cases if equal(a, b) != want]
    print(f"reference solutions: {len(items) - len(bad)} / {len(items)} match their answers (misses: {bad})")
    print(f"known forms: {len(cases) - len(wrong)} / {len(cases)} as expected (wrong: {wrong})")


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
