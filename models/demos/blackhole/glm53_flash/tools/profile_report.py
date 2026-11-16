"""Summarize a device profile of the decode step and the prefill chunk (tests/test_model_device_profile.py).

    python tools/profile_report.py <TT_METAL_PROFILER_DIR> [--json summary.json] [--top 12]

Run the test with TTNN_OP_PROFILER=1 (op names) and TT_METAL_PROFILER_MID_RUN_DUMP=1 (the report is written at every
read, so the row counts in stages.json split it).  The report has one row per program and chip.  A program counts with its slowest chip's kernel time (the mesh
moves at that chip's pace).  Per batch, it prints:
  - eager: each stage's (embed, every layer, head) programs and kernel time, and the ops that take the most time per
    kind of layer;
  - traced: the replayed step's kernel time and the gaps between programs, per stage (the trace runs the programs in
    the eager step's order, so the eager stages' program counts split it).
Per prefill chunk length: the chunk's host-clock time (and the time the host took to issue it), then the same per
stage tables as eager.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

KERNEL = "DEVICE KERNEL DURATION [ns]"
GAP = "OP TO OP LATENCY [ns]"


def number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return 0.0


def programs(header: list, lines: list) -> list:
    """Report lines (one per program and chip) -> one dict per program, in program order."""
    col = {name: i for i, name in enumerate(header)}
    by_uid = defaultdict(list)
    for line in lines:
        if len(line) < len(header) or line[0] == header[0]:
            continue
        # GLOBAL CALL COUNT = (the program's runtime id << 10) | chip id
        uid = (line[col["METAL TRACE REPLAY SESSION ID"]], int(line[col["GLOBAL CALL COUNT"]]) >> 10)
        by_uid[uid].append(line)
    out = []
    for uid in sorted(by_uid, key=lambda u: (u[0], u[1])):
        group = by_uid[uid]
        kernels = [number(x[col[KERNEL]]) for x in group]
        gaps = [number(x[col[GAP]]) for x in group]
        out.append(
            {
                "op": group[0][col["OP NAME"]],
                "chips": len(group),
                "cores": int(number(group[0][col["CORE COUNT"]])),
                "kernel_ns": max(kernels),
                "kernel_mean_ns": sum(kernels) / len(kernels),
                "gap_ns": sum(gaps) / len(gaps),
            }
        )
    return out


def by_op(progs: list) -> list:
    acc = defaultdict(lambda: [0, 0.0])
    for p in progs:
        acc[p["op"]][0] += 1
        acc[p["op"]][1] += p["kernel_ns"]
    return sorted(([op, n, ns] for op, (n, ns) in acc.items()), key=lambda r: -r[2])


PARTS = [
    "attn_hc.read",
    "input_norm",
    "attention",
    "attn_hc.write",
    "ffn_hc.read",
    "post_attention_norm",
    "mlp",
    "ffn_hc.write",
]


def split(progs: list, counts: list) -> dict:
    """The programs in order -> {stage: programs}, by the eager step's per-stage counts; the rest go to "head"."""
    out, i = {}, 0
    for stage, n in counts:
        out[stage] = progs[i : i + n]
        i += n
    out["head"] = out.get("head", []) + progs[i:]
    return out


def op_table(progs: list, title: str, top: int = 30) -> list:
    """Print and return the programs' totals by op (count, kernel ms of the slowest chip), largest first."""
    rows = by_op(progs)
    print(f"\n{title}:\n\n| op | programs | kernel ms |\n|---|---|---|")
    for op, n, ns in rows[:top]:
        print(f"| {op} | {n} | {ns / 1e6:.3f} |")
    return [[op, n, round(ns / 1e6, 4)] for op, n, ns in rows]


def ms(progs: list, key: str = "kernel_ns") -> float:
    return sum(p[key] for p in progs) / 1e6


def tables(staged: dict, kinds: list, top: int, with_ops: bool) -> dict:
    """Per layer totals, then parts x (first layer of each kind), then the ops of each part."""
    layers = sorted({int(k.split(" ", 1)[0]) for k in staged if k[0].isdigit()})
    print("| stage | kind | programs | kernel ms | gaps ms |")
    print("|---|---|---|---|---|")
    res = {"layers": {}, "parts": {}}
    for name in ["embed"] + [str(i) for i in layers] + ["head"]:
        progs = (
            staged.get(name, [])
            if not name.isdigit()
            else [p for k, v in staged.items() if k.split(" ", 1)[0] == name for p in v]
        )
        kind = kinds[int(name)] if name.isdigit() else ""
        res["layers"][name] = {
            "kind": kind,
            "programs": len(progs),
            "kernel_ms": ms(progs),
            "gap_ms": ms(progs, "gap_ns"),
        }
        print(f"| {name} | {kind} | {len(progs)} | {ms(progs):.3f} | {ms(progs, 'gap_ns'):.3f} |")
    first = {}
    for i in layers:
        first.setdefault(kinds[i], i)
    print("\n| part | " + " | ".join(f"{k} (layer {i}) ms / programs" for k, i in first.items()) + " |")
    print("|---|" + "---|" * len(first))
    for part in PARTS:
        cells = []
        for kind, i in first.items():
            progs = staged.get(f"{i} {part}", [])
            res["parts"].setdefault(kind, {})[part] = {
                "programs": len(progs),
                "kernel_ms": ms(progs),
                "ops": by_op(progs),
            }
            cells.append(f"{ms(progs):.3f} / {len(progs)}")
        print(f"| {part} | " + " | ".join(cells) + " |")
    if with_ops:
        for kind, i in first.items():
            print(f"\nlayer {i} ({kind}): ops per part (programs, ms)\n")
            for part in PARTS:
                ops = res["parts"][kind][part]["ops"][:top]
                print(
                    f"- {part}: "
                    + ", ".join(f"{op.replace('DeviceOperation', '')} {n} / {ns / 1e6:.3f}" for op, n, ns in ops)
                )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prof_dir", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    meta = json.loads((args.prof_dir / "stages.json").read_text())
    with open(args.prof_dir / ".logs" / "cpp_device_perf_report.csv") as f:
        lines = list(csv.reader(f))
    header = [h.strip() for h in lines[0]]
    kinds = meta["layers"]
    print(f"tree {meta['tree']}, layers {kinds}")

    summary = {}
    stages = meta["stages"]
    for users in sorted({s["users"] for s in stages if s["mode"] in ("eager", "trace")}):
        eager = [s for s in stages if s.get("users") == users and s["mode"] == "eager"]
        staged, counts = {}, []
        for prev, cur in zip(eager, eager[1:]):
            progs = programs(header, lines[prev["rows"] : cur["rows"]])
            staged[cur["stage"]] = progs
            counts.append((cur["stage"], len(progs)))
        print(
            f"\n## {users} user(s) per row ({users * 8} in all)\n\n### eager (kernel time of the slowest chip per program)\n"
        )
        res = {"eager": tables(staged, kinds, args.top, with_ops=True) if staged else None, "trace": None}

        traced = [s for s in stages if s.get("users") == users and s["mode"] == "trace"]
        if len(traced) == 2:
            progs = programs(header, lines[traced[0]["rows"] : traced[1]["rows"]])
            print(
                f"\n### traced step: {traced[1]['seconds'] * 1e3:.1f} ms on the host clock, {len(progs)} programs, "
                f"kernel {ms(progs):.1f} ms + gaps {ms(progs, 'gap_ns'):.1f} ms\n"
            )
            print("(split by the eager step's program counts; the head is the greedy one, so it has more programs)\n")
            res["trace"] = {"seconds": traced[1]["seconds"], "programs": len(progs), "kernel_ms": ms(progs)}
            if counts:  # (the stages: from the eager step's program counts)
                res["trace"].update(tables(split(progs, counts), kinds, args.top, with_ops=False))
            res["trace"]["by_op"] = op_table(progs, f"the traced step at {users} user(s) a row, by op")
        summary[users] = res

    for users in sorted({s["users"] for s in stages if s["mode"] == "spec"}):  # MTP's verify step
        spec = [s for s in stages if s["users"] == users and s["mode"] == "spec"]
        progs = programs(header, lines[spec[0]["rows"] : spec[1]["rows"]])
        print(
            f"\n## MTP's verify step, {users} user(s) a row ({2 * users} tokens): {spec[1]['seconds'] * 1e3:.1f} ms on the "
            f"host clock, {len(progs)} programs, kernel {ms(progs):.1f} ms + gaps {ms(progs, 'gap_ns'):.1f} ms"
        )
        summary[f"spec_{users}"] = {
            "seconds": spec[1]["seconds"],
            "programs": len(progs),
            "kernel_ms": ms(progs),
            "by_op": op_table(progs, "by op"),
        }

    for users, n in sorted({(s["users"], s["tokens"]) for s in stages if s["mode"] == "verify"}):  # forward_verify
        ver = [s for s in stages if s["mode"] == "verify" and s["users"] == users and s["tokens"] == n]
        progs = programs(header, lines[ver[0]["rows"] : ver[1]["rows"]])
        print(
            f"\n## the verify step, {users} user(s) a row x {n} tokens ({n * users} a row): {ver[1]['seconds'] * 1e3:.1f} ms "
            f"on the host clock, {len(progs)} programs, kernel {ms(progs):.1f} ms + gaps {ms(progs, 'gap_ns'):.1f} ms"
        )
        summary[f"verify_{users}x{n}"] = {
            "seconds": ver[1]["seconds"],
            "programs": len(progs),
            "kernel_ms": ms(progs),
            "by_op": op_table(progs, "by op", top=25),
        }

    # prefill (one request per row) and prefill_sp (one request's wave over the rows), per chunk length
    for mode, chunk in sorted({(s["mode"], s["chunk"]) for s in stages if s["mode"].startswith("prefill")}):
        pre = [s for s in stages if s["mode"] == mode and s["chunk"] == chunk]
        staged = {cur["stage"]: programs(header, lines[prev["rows"] : cur["rows"]]) for prev, cur in zip(pre, pre[1:])}
        what = "one request's wave over the rows" if mode == "prefill_sp" else "a chunk on every row"
        print(
            f"\n## {mode}, {what}, segments of {chunk}: {pre[0]['seconds'] * 1e3:.1f} ms on the host clock "
            f"(issued in {pre[0]['enqueued'] * 1e3:.1f} ms; profiler on)\n"
        )
        res = {"seconds": pre[0]["seconds"], "enqueued": pre[0]["enqueued"]}
        res.update(tables(staged, kinds, args.top, with_ops=True))
        allp = [p for v in staged.values() for p in v]
        res["by_op"] = op_table(allp, f"{mode} {chunk}: all stages by op", top=40)
        summary[f"{mode}_{chunk}"] = res

    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
