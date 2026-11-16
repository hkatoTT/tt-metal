"""A continuation file for the task-accuracy test (tests/test_model_device_tasks.py): the problems an earlier run left
unfinished, each with the tokens generated so far as its "prev_gen", so that another run prefills them after the prompt
and goes on (a generation longer than one run's time, e.g. AIME24 up to 131K tokens).

Taken from each results file (`tasks-*.json`, a form's record): its "unfinished" (stopped at the time budget) and its
"generations" that ended at the cap without an end token; problems given with --fresh start again from nothing (an
earlier run that did not save their tokens).  The items keep the tasks file's fields (prompt_ids, answer, ...).

    python -m models.demos.blackhole.glm53_flash.tools.tasks_continue --tasks TASKS.json --out OUT.json \
        [--results RUN.json ...] [--mode dflash7] [--fresh 13,28]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EOS = {154820, 154827, 154829}  # (tests/test_model_device_tasks.py's)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True, help="the tasks file the runs used (aime24.json, ...)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--results", type=Path, nargs="*", default=[], help="earlier runs' tasks-*.json, in order")
    p.add_argument("--mode", default="dflash7", help="the form whose generations to continue")
    p.add_argument("--fresh", default="", help="problem idx (the tasks file's 'idx') to start again from nothing")
    args = p.parse_args()
    doc = json.loads(args.tasks.read_text())
    by_idx = {x["idx"]: x for x in doc["items"]}
    prev: dict[int, list] = {}
    for path in args.results:
        run = json.loads(path.read_text())
        ids = run["problem_ids"]
        rec = next(r for r in run["results"] if r["mode"] == args.mode)
        for k, seq in rec.get("generations", {}).items():
            idx = ids[int(k)]
            if any(t in EOS for t in seq):
                prev.pop(idx, None)  # (finished: nothing to continue)
            else:
                prev[idx] = seq  # (ended at the cap: go on past it)
        for k, seq in rec.get("unfinished", {}).items():
            prev[ids[int(k)]] = seq
    for idx in (int(x) for x in args.fresh.split(",") if x):
        prev.setdefault(idx, [])
    items = [dict(by_idx[idx], prev_gen=seq) for idx, seq in sorted(prev.items())]
    out = dict(doc, items=items, version=doc.get("version", "") + "+continue")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print(
        f"wrote {args.out}: {len(items)} problems; tokens so far "
        + ", ".join(f"{x['idx']}: {len(x['prev_gen'])}" for x in items)
    )


if __name__ == "__main__":
    main()
