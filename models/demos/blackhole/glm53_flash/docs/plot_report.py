# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The figures of docs/BRINGUP_AND_OPTIMIZATION.md, from the measured values written below.

    python models/demos/blackhole/glm53_flash/docs/plot_report.py   # needs matplotlib; writes the PNGs into docs/report/

Every value is a measurement taken while the work was done (the report says what each figure shows and under which
conditions); nothing is recomputed here except unit conversions (per-wave times to per-1,000-token times, tok/s gains
to time savings).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = Path(__file__).parent / "report"
OUT.mkdir(exist_ok=True)

# chart surface and ink (as docs/perf/plot_perf.py)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# categorical slots, in this fixed order (validated for color-vision deficiencies on adjacent series)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# an ordinal blue ramp, light to dark
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
NEUTRAL_DARK, NEUTRAL_LIGHT = "#6f6e69", "#c9c8bf"  # "drafter" and "other": neutrals, not a sixth hue

# one color per part of the model, the same in every figure
PART = {"experts": CAT[0], "kda": CAT[1], "mhc": CAT[2], "dsa": CAT[3], "comm": CAT[4],
        "drafter": NEUTRAL_DARK, "other": NEUTRAL_LIGHT}  # fmt: skip

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "legend.frameon": False,
    }
)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def fmt_int(v):
    return f"{v:,.0f}"


# --------------------------------------------------------------------------------------------------------------------
# 1. Headline: the first end-to-end version against the release
# --------------------------------------------------------------------------------------------------------------------
def headline():
    rows = [  # (label, first end-to-end version, release)
        ("Decode, 8 users", 59.3, 1208),
        ("Decode, 32 users", 226.3, 3754),
        ("Decode, 128 users", 722.2, 8469),
        ("Prefill, 8 prompts x 2,048", 398, 30080),
        ("Prefill, one long prompt\n(2,048 then, 8K now)", 398, 24080),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    for i, (label, a, b) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([a, b], [y, y], color=AXIS, linewidth=2, solid_capstyle="round", zorder=1)
        ax.scatter([a], [y], s=60, color=NEUTRAL_LIGHT, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.scatter([b], [y], s=60, color=CAT[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(fmt_int(a), (a, y), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8.5, color=INK2)  # fmt: skip
        ax.annotate(f"{fmt_int(b)}  (x{b / a:.0f})", (b, y), xytext=(8, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=8.5, color=INK)  # fmt: skip
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.set_xlim(20, 250000)
    ax.set_xticks([30, 100, 300, 1000, 3000, 10000, 30000, 100000], ["30", "100", "300", "1K", "3K", "10K", "30K", "100K"])
    ax.minorticks_off()
    ax.set_xlabel("total tokens per second (log scale)")
    ax.set_title("From the first end-to-end version to the release", pad=26)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=NEUTRAL_LIGHT, markersize=7),
               plt.Line2D([], [], marker="o", linestyle="", color=CAT[0], markersize=7)]  # fmt: skip
    ax.legend(handles, ["first end-to-end version (BF16 weights, prefill one request at a time)", "release"],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8.5)  # fmt: skip
    fig.tight_layout()
    save(fig, "headline.png")


# --------------------------------------------------------------------------------------------------------------------
# 2. Decode tok/s at the fixed points, phase by phase
# --------------------------------------------------------------------------------------------------------------------
DECODE_POINTS = [  # label, 8 / 32 / 128 users total tok/s
    ("eager\nmodel", 6.7, 21.9, 50.8),
    ("traced\nstep", 32.7, 119.9, 314.7),
    ("first e2e\nversion", 59.3, 226.3, 722.2),
    ("BFP8\nweights", 69, 263, 833),
    ("first\ncustom ops", 153, 486, 1357),
    ("experts\nop", 174, 568, 1512),
    ("experts by\ncolumn", 245, 796, 2171),
    ("collectives,\nshared exp.", 324, 1001, 2444),
    ("plain\ndecode end", 380, 1124, 2565),
    ("MTP", 614, 1543, 4001),
    ("MTP\ntuned", 685, 1827, 4823),
    ("DFlash2", 941, 2596, 5593),
    ("verify step,\nfirst\nround", 1034, 3093, 6912),
    ("MoE way\nback in\nthe op, ...", 1093, 3313, 7310),
    ("L1 landing,\n2-stage\nAR, ...", 1199, 3755, 8432),
    ("release", 1208, 3754, 8469),
]
PHASES = [  # (first index, last index, name)
    (0, 2, "bring-up"),
    (3, 3, "BFP8"),
    (4, 8, "custom ops"),
    (9, 11, "speculative decoding"),
    (12, 15, "fixed time of the verify step"),
]


def decode_timeline():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    xs = list(range(len(DECODE_POINTS)))
    for lo, hi, name in PHASES:
        ax.axvspan(lo - 0.4, hi + 0.4, color=GRID, alpha=0.35, linewidth=0, zorder=0)
        ax.text((lo + hi) / 2, 25000, name, ha="center", va="center", fontsize=8.5, color=INK2)
    ax.axvline(8.5, color=INK2, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.text(8.6, 3.5, "from here: accepted tokens\n(drafts verified in one step)", fontsize=8, color=INK2, va="bottom")
    for j, (users, color) in enumerate(((8, ORD[0]), (32, ORD[2]), (128, ORD[4]))):
        ys = [p[1 + j] for p in DECODE_POINTS]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=f"{users} users", zorder=3)  # fmt: skip
        ax.annotate(f"{users} users: {fmt_int(ys[-1])}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)  # fmt: skip
        ax.annotate(f"{ys[0]:g}", (xs[0], ys[0]), xytext=(-6, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=8, color=INK2)  # fmt: skip
    ax.set_yscale("log")
    ax.set_yticks([3, 10, 30, 100, 300, 1000, 3000, 10000], ["3", "10", "30", "100", "300", "1K", "3K", "10K"])
    ax.minorticks_off()
    ax.set_ylim(2.5, 40000)
    ax.set_xlim(-0.8, len(xs) + 1.6)
    ax.set_xticks(xs, [p[0] for p in DECODE_POINTS], fontsize=7.8)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("decode, total tok/s (log scale)")
    ax.set_title("Decode throughput at the fixed measurement points, step by step")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3, bbox_to_anchor=(0.0, 0.92))
    fig.tight_layout()
    save(fig, "decode_timeline.png")


# --------------------------------------------------------------------------------------------------------------------
# 3. Waterfalls of the custom-op phase at 32 users
# --------------------------------------------------------------------------------------------------------------------
def waterfall(ax, start_label, start, steps, end_label, title, xmax):
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    labels = [start_label] + [s[0] for s in steps] + [end_label]
    n = len(labels)
    cur = start
    ax.barh(n - 1, start, height=0.6, color=NEUTRAL_LIGHT, linewidth=0)
    ax.text(start + xmax * 0.01, n - 1, f"{start:.1f} ms", va="center", fontsize=8.5, color=INK2)
    for i, (label, new) in enumerate(steps):
        y = n - 2 - i
        ax.barh(y, cur - new, left=new, height=0.6, color=CAT[0], linewidth=0)
        ax.text(cur + xmax * 0.01, y, f"-{cur - new:.1f}", va="center", fontsize=8.5, color=INK2)
        cur = new
    ax.barh(0, cur, height=0.6, color=CAT[0], linewidth=0, alpha=0.55)
    ax.text(cur + xmax * 0.01, 0, f"{cur:.1f} ms", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(n), list(reversed(labels)), fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("decode step at 32 users (ms)")
    ax.set_title(title, fontsize=10)


def custom_ops_waterfall():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1, 1.35]})
    waterfall(
        a, "model code + Sinkhorn op", 81.3,
        [("KDA recurrence op", 77.2), ("mHC write op", 72.3), ("mHC read ops", 64.5), ("KDA prepare op", 58.6),
         ("KDA finish op", 58.2), ("MoE routing and plan op", 53.9)],
        "first custom ops", "(a) First custom ops (random weights)", 95,
    )  # fmt: skip
    waterfall(
        b, "first custom ops", 65.9,
        [("experts op", 56.3), ("Sinkhorn on the SFPU; h by multicast", 53.8),
         ("KDA / mHC kernels: fewer reads and waits", 48.8),
         ("mHC read + norm op, DSA matmul configs,\nsmall collectives row-major", 45.3),
         ("experts split by mesh column", 40.2), ("LM head over the mesh; experts op fixes", 38.0),
         ("KDA ops: SFPU on live rows only", 36.3), ("shared expert inside the experts op;\nmHC read regrouped", 35.2),
         ("MoE return as column all-reduce;\nasync row all-reduce; mHC", 32.0),
         ("decode_matmul; one-shot row all-reduce", 30.5),
         ("experts op fixed time; routing a row;\none-shot column gather", 29.2),
         ("custom column scatter", 28.5)],
        "end of plain decode", "(b) Then with the real weights", 78,
    )  # fmt: skip
    fig.suptitle("Where the custom-op phase's time went (32 users, plain decode step, each change in order)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")  # fmt: skip
    fig.tight_layout(w_pad=3)
    save(fig, "custom_ops_waterfall.png")


# --------------------------------------------------------------------------------------------------------------------
# 4. One expert in the decode MoE
# --------------------------------------------------------------------------------------------------------------------
def expert_op():
    rows = [
        ("sparse_matmul x 2, 64 cores x 2 columns", 99, NEUTRAL_LIGHT),
        ("sparse_matmul x 2, 32 cores x 4 columns", 87, NEUTRAL_LIGHT),
        ("experts op, first version", 145, CAT[0]),
        ("+ weight reads on NoC 0", 132, CAT[0]),
        ("+ h by unicast, down cores next to gate/up", 82, CAT[0]),
        ("+ each core's weights in one DRAM bank", 74, CAT[0]),
        ("experts op, weight reads only", 65, CAT[0]),
    ]
    bound = 26.7e6 / 512e9 * 1e6
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    for i, (label, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.6, color=c, linewidth=0, alpha=0.55 if "only" in label else 1.0)
        ax.text(v + 1.5, y, f"{v} µs ({26.7e6 / (v * 1e-6) / 1e9:.0f} GB/s)", va="center", fontsize=8.5, color=INK2)
    ax.axvline(bound, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(bound + 1, len(rows) - 0.45, f"26.7 MB at 512 GB/s: {bound:.0f} µs", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 185)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("one expert on one chip, µs (gate, up, down: 26.7 MB in BFP8)")
    ax.set_title("Decode MoE: one expert's time, from ttnn ops to a custom op")
    fig.tight_layout()
    save(fig, "expert_op.png")


# --------------------------------------------------------------------------------------------------------------------
# helper: stacked horizontal bars by part
# --------------------------------------------------------------------------------------------------------------------
def stacked(ax, rows, parts, names, height=0.6, label_min=2.0, unit="", first_label_left=False):
    """rows: list of (label, {part: value}); parts in stacking order.  first_label_left: write the first part's value
    at its left end (clear of a bound tick drawn across the bar) instead of its middle."""
    for i, (label, vals) in enumerate(rows):
        y = len(rows) - 1 - i
        left = 0.0
        for p in parts:
            v = vals.get(p, 0.0)
            if v <= 0:
                continue
            ax.barh(y, v, left=left, height=height, color=PART[p], edgecolor=SURFACE, linewidth=1.5)
            if v >= label_min:
                tc = "white" if p in ("experts", "drafter") else INK
                if first_label_left and left == 0.0:
                    ax.text(left + 1.5, y, f"{v:.1f}", ha="left", va="center", fontsize=7.5, color=tc, zorder=5,
                            bbox={"facecolor": PART[p], "edgecolor": "none", "pad": 1.0})
                else:
                    ax.text(left + v / 2, y, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color=tc)
            left += v
        ax.text(left + 0.6, y, f"{left:.1f}{unit}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(color=PART[p]) for p in parts]
    return handles, [names[p] for p in parts]


# --------------------------------------------------------------------------------------------------------------------
# 5. Plain decode step by part: BFP8 baseline, end of the custom-op phase, layout bound
# --------------------------------------------------------------------------------------------------------------------
def plain_step_parts():
    parts = ["experts", "kda", "mhc", "dsa", "other"]
    names = {"experts": "MoE (experts, communication, router)", "kda": "KDA", "mhc": "mHC", "dsa": "DSA",
             "other": "other (dense MLP, LM head, embedding)"}  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 2.6), sharex=True)
    data = {
        8: ([("BFP8 weights, model code only", {"experts": 61.9, "kda": 20.1, "mhc": 22.4, "dsa": 3.9, "other": 8.1}),
             ("end of the custom-op phase", {"experts": 17.2, "kda": 5.7, "mhc": 2.6, "dsa": 2.3, "other": 0.9})],
            9.5),
        128: ([("BFP8 weights, model code only", {"experts": 79.1, "kda": 36.1, "mhc": 27.8, "dsa": 6.0, "other": 8.1}),
               ("end of the custom-op phase", {"experts": 37.4, "kda": 10.6, "mhc": 4.0, "dsa": 2.9, "other": 1.2})],
              26.8),
    }  # fmt: skip
    for ax, users in zip(axes, (8, 128)):
        rows, bound = data[users]
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        handles, labels = stacked(ax, rows, parts, names, label_min=12.0, first_label_left=True)
        ax.plot([bound, bound], [-0.45, 0.45], color=INK, linewidth=1.4)
        ax.annotate(f"layout bound {bound} ms", (bound, -0.45), xytext=(0, -12), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)  # fmt: skip
        ax.set_title(f"{users} users", fontsize=10)
        ax.set_xlabel("ms a step (sum over ops of the slowest chip, random weights)")
        ax.set_ylim(-0.9, 1.5)
    axes[0].set_xlim(0, 175)
    fig.legend(handles, labels, ncol=5, loc="upper left", bbox_to_anchor=(0.01, 1.02), fontsize=8.5)
    fig.suptitle("Plain decode step by part, before and after the custom ops", x=0.01, y=1.09, ha="left",
                 fontsize=11, fontweight="bold")  # fmt: skip
    fig.tight_layout(w_pad=3)
    save(fig, "plain_step_parts.png")


# --------------------------------------------------------------------------------------------------------------------
# 6. Speculative decoding: tokens a step and the choice of K
# --------------------------------------------------------------------------------------------------------------------
def spec_k():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [0.85, 1, 1]})
    # (a) tokens a step (CPU reference, the 11 records to their end) with 95 % intervals
    ks = ["MTP\n(1 draft)", "K = 3", "K = 4", "K = 5", "K = 6", "K = 7"]
    tok = [1.892, 3.118, 3.503, 3.776, 4.092, 4.182]
    lo = [1.86, 2.87, 3.12, 3.31, 3.55, 3.63]
    hi = [1.93, 3.38, 3.91, 4.26, 4.68, 4.77]
    for i, (t, l, h) in enumerate(zip(tok, lo, hi)):
        color = CAT[1] if i == 0 else CAT[0]
        a.plot([i, i], [l, h], color=color, linewidth=2, solid_capstyle="round", alpha=0.45)
        a.scatter([i], [t], s=45, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        a.annotate(f"{t:.2f}", (i, t), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8,
                   color=INK2)  # fmt: skip
    a.set_xticks(range(len(ks)), ks, fontsize=8.5)
    a.set_xlim(-0.5, len(ks) - 0.3)
    a.set_ylim(0, 5.2)
    a.set_ylabel("tokens a user advances in one step")
    a.set_title("(a) Tokens a step on real text", fontsize=10)
    # (b) the first DFlash2 measurement (real weights): total tok/s by users and K
    users = [8, 16, 32, 64, 128]
    first = {
        "MTP (1 draft)": [689, 1158, 1909, 3073, 5240],
        "DFlash2 K = 3": [871, 1444, 2358, 4033, 5593],
        "DFlash2 K = 7": [941, 1503, 2596, 3634, 3991],
    }
    colors = {"MTP (1 draft)": CAT[1], "DFlash2 K = 3": ORD[1], "DFlash2 K = 7": ORD[4]}
    for name, vals in first.items():
        b.plot(users, vals, color=colors[name], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
               markeredgewidth=1.5, label=name)  # fmt: skip
    b.set_title("(b) First DFlash2 on the device", fontsize=10)
    # (c) the release (PERF.md): K = 7 and K = 3 where both were measured
    rel7 = {8: 1208, 16: 1881, 32: 3754, 64: 5718}
    rel3 = {16: 1758, 64: 5332, 128: 8469}
    c.plot(list(rel3), list(rel3.values()), color=ORD[1], linewidth=1.2, linestyle=(0, (3, 2)), marker="o",
           markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label="DFlash2 K = 3")  # fmt: skip
    c.plot(list(rel7), list(rel7.values()), color=ORD[4], linewidth=1.2, linestyle=(0, (3, 2)), marker="o",
           markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label="DFlash2 K = 7")  # fmt: skip
    for u in (16, 64):
        c.annotate(f"K=7 {fmt_int(rel7[u])}\nK=3 {fmt_int(rel3[u])}", (u, rel7[u]), xytext=(-6, 10),
                   textcoords="offset points", ha="right", fontsize=7.5, color=INK2)  # fmt: skip
    c.set_title("(c) Release: K = 7 now ahead up to 64 users", fontsize=10)
    for ax in (b, c):
        ax.set_xscale("log", base=2)
        ax.set_xticks(users, [str(u) for u in users])
        ax.minorticks_off()
        ax.set_xlabel("users")
        ax.set_ylabel("total tok/s")
        ax.set_ylim(0, 9500)
        ax.set_yticks([0, 2000, 4000, 6000, 8000], ["0", "2,000", "4,000", "6,000", "8,000"])
        ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout(w_pad=2.5)
    save(fig, "spec_k.png")


# --------------------------------------------------------------------------------------------------------------------
# 7. The release's DFlash2 step by part (random weights profile), default and preset C
# --------------------------------------------------------------------------------------------------------------------
def verify_parts():
    parts = ["experts", "comm", "kda", "mhc", "dsa", "drafter", "other"]
    names = {"experts": "experts op", "comm": "router, routing, collectives", "kda": "KDA (matmuls and state ops)",
             "mhc": "mHC", "dsa": "DSA", "drafter": "drafter (8 layers, head)",
             "other": "other (dense MLP, heads, acceptance)"}  # fmt: skip
    rows = [
        ("32 users, default", {"experts": 22.52, "comm": 1.02 + 2.20, "kda": 3.16 + 3.20, "mhc": 2.97, "dsa": 1.92,
                               "drafter": 2.52 + 0.38, "other": 0.40 + 0.36 + 0.33}),
        ("32 users, preset C", {"experts": 13.94, "comm": 1.02 + 1.71, "kda": 3.07 + 3.20, "mhc": 2.78, "dsa": 1.90,
                                "drafter": 2.54 + 0.40, "other": 0.39 + 0.34 + 0.33}),
        ("128 users, default", {"experts": 26.45, "comm": 1.10 + 2.72, "kda": 3.39 + 5.93, "mhc": 4.01, "dsa": 2.42,
                                "drafter": 3.23 + 0.65, "other": 0.46 + 0.57 + 0.46}),
        ("128 users, preset C", {"experts": 17.83, "comm": 1.09 + 2.21, "kda": 3.38 + 5.93, "mhc": 4.10, "dsa": 2.41,
                                 "drafter": 3.22 + 0.64, "other": 0.46 + 0.56 + 0.46}),
    ]  # fmt: skip
    bounds = [20.5, 11.3, 21.4, 12.1]
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    handles, labels = stacked(ax, rows, parts, names, label_min=2.2)
    for i, bd in enumerate(bounds):
        y = len(rows) - 1 - i
        ax.plot([bd, bd], [y - 0.38, y + 0.38], color=INK, linewidth=1.6)
    ax.plot([], [], color=INK, linewidth=1.6, label="bound")
    handles.append(plt.Line2D([], [], color=INK, linewidth=1.6))
    labels.append("verify step's weight-read bound")
    ax.set_xlim(0, 58)
    ax.set_xlabel("ms a step (sum over ops of the slowest chip, random weights, DFlash2 K = 7 at 32 users, K = 3 at 128)")
    ax.legend(handles, labels, ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.32), fontsize=8.5)
    ax.set_title("The decode step just before the release, by part", pad=58)
    fig.tight_layout()
    save(fig, "verify_parts.png")


# --------------------------------------------------------------------------------------------------------------------
# 8. Measured gains of the main techniques of the speculative-decoding phase
# --------------------------------------------------------------------------------------------------------------------
def step_gains():
    rows = [  # label, ms saved at 32 users, at 128 users (None: not measured; a string: why not), real weights?
        ("Experts op: an expert's weights read once\nper step, not once per token group", None, 4.63, True),
        ("KDA verify: one chunk from the last step's\nstate instead of K checkpoints", 3.12, 3.23, False),
        ("KDA prepare: a user's tokens in one tile", 1.75, "not used at 128 users", False),
        ("Row all-reduce at 64 tokens a row: async,\ninput and output in L1", None, 2.0, False),
        ("L1 landing of the next MoE layer's experts", 1.15, 0.82, True),
        ("Column gather inside the experts op", 0.46, 1.43, False),
        ("Column reduce-scatter inside the experts op", 0.75, 0.98, True),
        ("mHC write and next read in one op", 0.43, 0.95, False),
        ("Experts op writes its rows on NoC 0", 0.44, 0.95, False),
        ("Two-stage row all-reduce", 0.57, 0.85, True),
        ("DSA attention with the heads first\n(no transposes around it)", 0.34, 0.59, False),
        ("Kernel-config ring planned over the trace\n(+ drafter op fusions)", 0.29, 0.39, True),
        ("Fabric routes made on the host", 0.21, 0.35, False),
        ("KDA gates computed inside x_proj's matmul", 0.23, 0.30, False),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    h = 0.36
    for i, (label, v32, v128, real) in enumerate(rows):
        y = len(rows) - 1 - i
        for off, v, color in ((h / 2, v32, ORD[1]), (-h / 2, v128, ORD[4])):
            if v is None or isinstance(v, str):
                ax.text(0.04, y + off, v or "not measured", va="center", fontsize=7.5, color=MUTED)
                continue
            ax.barh(y + off, v, height=h, color=color, edgecolor=SURFACE, linewidth=1.5,
                    hatch=None if real else "////")  # fmt: skip
            ax.text(v + 0.05, y + off, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 5.2)
    ax.set_xlabel("ms saved per decode step (MTP or DFlash2 verify step of the time)")
    handles = [Patch(color=ORD[1]), Patch(color=ORD[4]), Patch(facecolor=SURFACE, edgecolor=INK2, hatch="////")]
    ax.legend(handles, ["32 users", "128 users", "hatched: measured with random weights"], loc="lower right",
              fontsize=8.5)  # fmt: skip
    ax.set_title("Measured gains of the main decode techniques after speculative decoding")
    fig.tight_layout()
    save(fig, "step_gains.png")


# --------------------------------------------------------------------------------------------------------------------
# 9. Teacher-forced KLD along the way
# --------------------------------------------------------------------------------------------------------------------
def kld_history():
    labels = ["first e2e\n(BF16)", "BFP8\nweights", "end of\nmodel code", "experts\nop", "experts by\nmesh column",
              "KDA\nops", "async row\nall-reduce", "decode_\nmatmul", "end of plain\ndecode", "release\nera"]  # fmt: skip
    dec = [0.0020, 0.0025, 0.00292, 0.00267, 0.00259, 0.00254, 0.00236, 0.00228, 0.00268, 0.00242]
    pre = [0.0016, 0.0024, 0.00279, 0.00250, 0.00270, 0.00238, 0.00239, 0.00254, 0.00282, 0.00228]
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    xs = range(len(labels))
    ax.plot(xs, dec, color=CAT[0], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="against the CPU reference's decode path")  # fmt: skip
    ax.plot(xs, pre, color=CAT[1], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="against the CPU reference's prefill path")  # fmt: skip
    ax.set_xticks(list(xs), labels, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 0.0036)
    ax.set_yticks([0, 0.001, 0.002, 0.003], ["0", "0.001", "0.002", "0.003"])
    ax.set_ylabel("mean KLD, 979 positions\n(11 records on 16 seats)")
    ax.set_title("Teacher-forced KL divergence on the 11 reference records, phase by phase")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "kld_history.png")


# --------------------------------------------------------------------------------------------------------------------
# 10. Prefill throughput, phase by phase
# --------------------------------------------------------------------------------------------------------------------
def prefill_timeline():
    labels = ["first e2e\n(one at a time)", "waves of 8\n(one a row)", "BFP8\nweights", "custom\nops",
              "all chunks traced,\nMoE read once", "sequence\nparallel", "MoE, mHC,\nKDA prefill ops",
              "waves of 1,024\na row", "fused MoE\nway back, 2D gather", "mHC over a\nrow's chips", "release"]  # fmt: skip
    multi = [398, 3309, 3404, 5852, 15751, None, 20570, 24124, 27130, 28415, 30080]
    one2k = [398, None, 445, 788, 2503, 7699, None, None, None, None, None]
    one8k = [None, None, None, None, None, 12552, 17580, 18823, 21980, 23109, 24080]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    xs = list(range(len(labels)))
    for vals, color, name, marker, filled in ((multi, ORD[4], "8 prompts x 2,048", "o", True),
                                              (one8k, ORD[1], "one prompt of 8K", "o", True),
                                              (one2k, ORD[1], "one prompt of 2,048", "o", False)):  # fmt: skip
        pts = [(x, v) for x, v in zip(xs, vals) if v is not None]
        px, py = zip(*pts)
        ax.plot(px, py, color=color, linewidth=2 if filled else 1.4, linestyle="-" if filled else (0, (3, 2)),
                marker=marker, markersize=6, markerfacecolor=color if filled else SURFACE, markeredgecolor=color,
                markeredgewidth=1.5, label=name)  # fmt: skip
        ax.annotate(fmt_int(py[-1]), (px[-1], py[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK)  # fmt: skip
    ax.set_yscale("log")
    ax.set_yticks([300, 1000, 3000, 10000, 30000], ["300", "1K", "3K", "10K", "30K"])
    ax.minorticks_off()
    ax.set_ylim(250, 60000)
    ax.set_xlim(-0.5, len(xs) + 0.2)
    ax.set_xticks(xs, labels, fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("prefill, tok/s (log scale)")
    ax.set_title("Prefill throughput (random weights; eager wall time at first, traced from \"all chunks traced\" on)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "prefill_timeline.png")


# --------------------------------------------------------------------------------------------------------------------
# 11. Prefill device time per 1,000 prompt tokens, by part
# --------------------------------------------------------------------------------------------------------------------
def prefill_parts():
    parts = ["experts", "comm", "kda", "mhc", "dsa", "other"]
    names = {"experts": "MoE experts op", "comm": "MoE way back, router, shared expert", "kda": "KDA", "mhc": "mHC",
             "dsa": "DSA", "other": "other"}  # fmt: skip

    def per_k(tokens, d):
        return {k: v * 1000 / tokens for k, v in d.items()}

    rows = [
        ("8 x 2,048, 512 a row a wave\n(end of the custom-op phase)",
         per_k(4096, {"experts": 390, "comm": 100, "mhc": 90, "kda": 50, "dsa": 35})),
        ("8 x 2,048, 1,024 a row a wave\n(release)",
         per_k(8192, {"experts": 106.24, "comm": 55.99 + 9.23 + 8.50, "kda": 14.87 + 24.04 + 3.63 + 9.42 + 3.98,
                      "mhc": 10.45, "dsa": 3.21 + 15.16 + 2.28 + 9.17, "other": 4.23})),
        ("1 x 8K, sequence parallel\n(release)",
         per_k(8192, {"experts": 108.96, "comm": 57.47 + 8.63 + 8.52, "kda": 24.02 + 23.39 + 20.92 + 9.46 + 4.05,
                      "mhc": 10.74, "dsa": 23.32 + 18.28 + 14.56 + 12.70, "other": 4.24})),
    ]  # fmt: skip
    fig, ax = plt.subplots(figsize=(10, 2.9))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    handles, labels = stacked(ax, rows, parts, names, label_min=4.0, unit=" ms")
    ax.set_xlim(0, 185)
    ax.set_xlabel("device time per 1,000 prompt tokens (ms; sum over ops of the slowest chip, random weights)")
    ax.legend(handles, labels, ncol=6, loc="upper left", bbox_to_anchor=(0.0, 1.22), fontsize=8.5)
    ax.set_title("Prefill device time by part", pad=30)
    fig.tight_layout()
    save(fig, "prefill_parts.png")


# --------------------------------------------------------------------------------------------------------------------
# 12. Measured prefill gains (value-preserving unless marked)
# --------------------------------------------------------------------------------------------------------------------
def prefill_gains():
    def from_tok(a, b):
        return 100 * (1 - a / b)

    rows = [  # label, % of prefill time saved for 8 x 2,048, for 1 x 8K
        ("waves of 1,024 tokens a row (two 512 waves as one)", from_tok(21025, 24124), None),
        ("two sequence-parallel waves as one; unchanged\ntrace inputs not rewritten", None, from_tok(17845, 18823)),
        ("mHC streams split over a row's 4 chips", 5.2, 4.6),
        ("trace inputs packed in one write (new prompt)", None, 100 * (1 - 1 / 1.049)),
        ("larger K blocks in the prefill matmuls\n(changes values; lowers the KLD)", 2.7, 3.7),
        ("combine carried in the reduce-scatter program", 2.9, None),
        ("x gather: column trunk + row relay", 2.4, 2.0),
        ("MoE plan built in the gather program", 2.0, 1.8),
        ("column RS + add + row RS + row AG in one op", 1.9, 1.5),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    h = 0.36
    for i, (label, a, b) in enumerate(rows):
        y = len(rows) - 1 - i
        for off, v, color in ((h / 2, a, ORD[4]), (-h / 2, b, ORD[1])):
            if v is None:
                continue
            ax.barh(y + off, v, height=h, color=color, edgecolor=SURFACE, linewidth=1.5)
            ax.text(v + 0.12, y + off, f"{v:.1f} %", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 15)
    ax.set_xlabel("prefill time saved (%)")
    ax.legend([Patch(color=ORD[4]), Patch(color=ORD[1])], ["8 prompts x 2,048", "one prompt of 8K"],
              loc="lower right", fontsize=8.5)  # fmt: skip
    ax.set_title("Measured gains of single prefill changes (random weights, traced)")
    fig.tight_layout()
    save(fig, "prefill_gains.png")


# --------------------------------------------------------------------------------------------------------------------
# 13. Long contexts: cache bytes and the longest context a user can hold
# --------------------------------------------------------------------------------------------------------------------
def long_context():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1, 1.1]})
    rows = [
        ("first version: per-head K and V in bf16", 176.0),
        ("bf16 latents (+ indexer keys), every chip of a row", 11.7),
        ("FP8 latents, BFP4 indexer keys", (5632 + 198) / 1024),
        ("... and the latents split over the row's 4 chips", (5632 / 4 + 198) / 1024),
    ]
    a.grid(True, axis="x")
    a.grid(False, axis="y")
    for i, (label, v) in enumerate(rows):
        y = len(rows) - 1 - i
        a.plot([0.8, v], [y, y], color=AXIS, linewidth=2, solid_capstyle="round")
        a.scatter([v], [y], s=55, color=CAT[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        a.annotate(f"{v:.3g} KiB", (v, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5,
                   color=INK)  # fmt: skip
    a.set_xscale("log")
    a.set_xlim(0.8, 600)
    a.set_xticks([1, 3, 10, 30, 100, 300], ["1", "3", "10", "30", "100", "300"])
    a.minorticks_off()
    a.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8.5)
    a.tick_params(axis="y", length=0)
    a.set_xlabel("DSA cache per token, per user, per chip (KiB, log scale; 11 layers)")
    a.set_title("(a) Cache bytes a token", fontsize=10)
    users = [8, 16, 32, 64, 128]
    first = [2048, 2048, 2048, 2048, 2048]
    bf16 = [680, 448, 272, 152, 80]
    dflt = [1024, 1024, 1024, 1024, 576]
    b.axhline(1024, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    b.text(8, 1150, "model maximum: 1M tokens", fontsize=8, color=INK2)
    b.plot(users, [v / 1024 for v in first], color=NEUTRAL_LIGHT, linewidth=2, marker="o", markersize=5,
           markeredgecolor=SURFACE, markeredgewidth=1.5, label="first version (no indexer: 2,048 tokens)")  # fmt: skip
    b.plot(users, bf16, color=ORD[1], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
           markeredgewidth=1.5, label="bf16 latents and keys on every chip")  # fmt: skip
    b.plot(users, dflt, color=ORD[4], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
           markeredgewidth=1.5, label="default: FP8 / BFP4, split over a row")  # fmt: skip
    for u, v in zip(users, bf16):
        b.annotate(f"{v}K", (u, v), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    b.annotate("576K", (128, 576), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    b.set_xscale("log", base=2)
    b.set_xticks(users, [str(u) for u in users])
    b.set_yscale("log")
    b.set_yticks([2, 10, 30, 100, 300, 1000], ["2K", "10K", "30K", "100K", "300K", "1M"])
    b.minorticks_off()
    b.set_ylim(1.2, 3000)
    b.set_xlabel("users")
    b.set_ylabel("longest context a user (tokens, log scale)")
    b.set_title("(b) Longest context, all users at the same length", fontsize=10)
    b.legend(loc="lower left", bbox_to_anchor=(0.0, 0.12), fontsize=8)
    fig.tight_layout(w_pad=3)
    save(fig, "long_context.png")


# --------------------------------------------------------------------------------------------------------------------
# 14. Serving throughput, milestone by milestone
# --------------------------------------------------------------------------------------------------------------------
def serving_timeline():
    labels = ["plain decode", "MTP,\n1 draft", "DFlash2, prefill\none row a request", "+ sequence-\nparallel prefill",
              "+ first drafts\ntraced, faster step", "+ run-ahead\ndecode step", "vLLM 0.31\n(new prompts)",
              "+ gathered\nprefills", "release"]  # fmt: skip
    rnd = [898.3, 1316.8, 1203.4, 1515, 1926, 2240.3, 2085, 2676, 2648]
    wiki = [None, 1161.0, 1018.8, 1237, 1468, 1729.2, 1773, 1966, 2157]
    k8 = [526.9, 816.9, 338.3, 857, 929, None, 1095, None, 1107]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    xs = list(range(len(labels)))
    ax.axvline(5.5, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(5.55, 150, "benchmark prompts changed\n(vLLM 0.30 to 0.31)", fontsize=8, color=INK2, va="bottom")
    for vals, color, name in ((rnd, ORD[4], "1K random-id prompts"), (wiki, CAT[1], "about 1K, wikitext"),
                              (k8, ORD[1], "8K random-id prompts")):  # fmt: skip
        pts = [(x, v) for x, v in zip(xs, vals) if v is not None]
        px, py = zip(*pts)
        ax.plot(px, py, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=name)  # fmt: skip
        ax.annotate(fmt_int(py[-1]), (px[-1], py[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK)  # fmt: skip
    ax.set_ylim(0, 3000)
    ax.set_xlim(-0.5, len(xs) - 0.2)
    ax.set_xticks(xs, labels, fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("output tok/s")
    ax.set_title("Serving: 32 concurrent requests, 1,024 output tokens, real weights, greedy")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "serving_timeline.png")


# --------------------------------------------------------------------------------------------------------------------
# 15. Serving features: throughput gains and time to first token
# --------------------------------------------------------------------------------------------------------------------
def serving_gains():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1]})
    rows = [  # label, random 1K, wikitext 1K
        ("DFlash2 prefill split over the 8 rows", 25.9, 21.4),
        ("run-ahead decode step", 19.2, 13.4),
        ("gathered prefills, wait up to 1.5 s\n(on top of run-ahead at request ends)", 10.0, 7.7),
        ("gathered prefills, wait up to 200 ms\n(first try)", 1.8, 2.1),
    ]
    a.grid(True, axis="x")
    a.grid(False, axis="y")
    h = 0.36
    for i, (label, r, w) in enumerate(rows):
        y = len(rows) - 1 - i
        for off, v, color in ((h / 2, r, ORD[4]), (-h / 2, w, CAT[1])):
            a.barh(y + off, v, height=h, color=color, edgecolor=SURFACE, linewidth=1.5)
            a.text(v + 0.3, y + off, f"+{v:.1f} %", va="center", fontsize=7.5, color=INK2)
    a.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8.5)
    a.tick_params(axis="y", length=0)
    a.set_xlim(0, 32)
    a.set_xlabel("output tok/s gain (32 concurrent, 1K in, 1K out)")
    a.legend([Patch(color=ORD[4]), Patch(color=CAT[1])], ["1K random-id prompts", "about 1K, wikitext"],
             loc="lower right", fontsize=8.5)  # fmt: skip
    a.set_title("(a) Throughput", fontsize=10)
    ttft = [  # label, before, after (s)
        ("gathered prefills (cost): 32 concurrent, 1K random ids", 0.23, 0.94),
        ("adaptive gather stop: 4 concurrent, 128 in", 1.64, 0.20),
        ("adaptive gather stop: 4 concurrent, 1K in", 1.70, 0.35),
        ("adaptive gather stop: 4 concurrent, 8K in", 2.17, 0.46),
        ("adaptive gather stop: 8 concurrent, 8K in", 2.96, 0.85),
        ("prefix store: 2nd turn of ~30K-token chats", 4.20, 0.98),
        ("prefix store: 3rd turn", 5.69, 0.58),
        ("prefix store: 4th turn", 5.15, 0.55),
    ]
    b.grid(True, axis="x")
    b.grid(False, axis="y")
    for i, (label, x0, x1) in enumerate(ttft):
        y = len(ttft) - 1 - i
        b.plot([x1, x0], [y, y], color=AXIS, linewidth=2, solid_capstyle="round")
        b.scatter([x0], [y], s=50, color=NEUTRAL_LIGHT, edgecolor=SURFACE, linewidth=2, zorder=3)
        b.scatter([x1], [y], s=50, color=CAT[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        lo_first = x1 < x0
        b.annotate(f"{x0:g} s", (x0, y), xytext=(8 if lo_first else -8, 0), textcoords="offset points",
                   ha="left" if lo_first else "right", va="center", fontsize=8, color=INK2)  # fmt: skip
        b.annotate(f"{x1:g} s", (x1, y), xytext=(-8 if lo_first else 8, 0), textcoords="offset points",
                   ha="right" if lo_first else "left", va="center", fontsize=8, color=INK)  # fmt: skip
    b.set_xscale("log")
    b.set_xlim(0.08, 12)
    b.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10], ["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    b.minorticks_off()
    b.set_yticks(range(len(ttft)), [t[0] for t in reversed(ttft)], fontsize=8.5)
    b.tick_params(axis="y", length=0)
    b.set_xlabel("median time to first token (s, log scale)")
    b.set_ylim(-0.6, len(ttft) - 0.4)
    b.legend([plt.Line2D([], [], marker="o", linestyle="", color=NEUTRAL_LIGHT, markersize=7),
              plt.Line2D([], [], marker="o", linestyle="", color=CAT[0], markersize=7)], ["before", "after"],
             loc="upper right", fontsize=8.5)  # fmt: skip
    b.set_title("(b) Time to first token", fontsize=10)
    fig.suptitle("Serving features, measured with the real weights", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")  # fmt: skip
    fig.tight_layout(w_pad=3)
    save(fig, "serving_gains.png")


# --------------------------------------------------------------------------------------------------------------------
# 16. Start-up: the real weights' load time, step by step
# --------------------------------------------------------------------------------------------------------------------
def startup():
    rows = [
        ("convert to BFP8 on the host, layer by layer", 486.2),
        ("skip redundant conversion work", 438.3),
        ("pack block-float tiles in one pass (C++)", 344.0),
        ("reuse the experts' slice buffers", 301.9),
        ("read the checkpoint ahead on 4 threads", 281.6),
        ("drop finished layers' pages from the page cache", 194.2),
        ("pack expert shards straight from their slices", 177.5),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    for i, (label, v) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.6, color=CAT[0] if i else NEUTRAL_LIGHT, linewidth=0)
        ax.text(v + 4, y, f"{v:.0f} s", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 560)
    ax.set_xlabel("loading the real weights (s; each row adds one change to the row above)")
    ax.set_title("Start-up: the weight load, change by change (same bytes on the device every time)")
    fig.tight_layout()
    save(fig, "startup.png")


# --------------------------------------------------------------------------------------------------------------------
# 17. Fast presets: speed against quality
# --------------------------------------------------------------------------------------------------------------------
def presets_tradeoff():
    prog = [  # label, step reduction at 32 users (%, random weights), verify-step KLD (real weights)
        ("default (BFP8, HiFi4)", 0.0, 0.00361),
        ("F = 0.8 (HiFi3)", 2.9, 0.00428),
        ("F = 0.7 (A)", 10.9, 0.00483),
        ("F = 0.6 (B)", 16.4, 0.00552),
        ("F = 0.55", 19.5, 0.00958),
        ("F = 0.5 (C)", 21.6, 0.01466),
    ]
    bfp4 = ("all-BFP4", 22.3, 0.0283)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.grid(True, axis="both")
    xs = [p[1] for p in prog]
    ys = [p[2] for p in prog]
    ax.plot(xs, ys, color=CAT[0], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="progressive form (BFP4 base + BFP4 correction)")  # fmt: skip
    ax.scatter([bfp4[1]], [bfp4[2]], s=60, color=CAT[1], edgecolor=SURFACE, linewidth=2, zorder=3,
               label="all experts in BFP4")  # fmt: skip
    nudge = {"default (BFP8, HiFi4)": (8, -2), "F = 0.8 (HiFi3)": (8, 6), "F = 0.7 (A)": (-6, 10),
             "F = 0.6 (B)": (8, -6), "F = 0.55": (-8, 8), "F = 0.5 (C)": (-8, 8)}  # fmt: skip
    for label, x, y in prog:
        dx, dy = nudge[label]
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK2,
                    ha="left" if dx > 0 else "right")  # fmt: skip
    ax.annotate(bfp4[0], (bfp4[1], bfp4[2]), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                fontsize=8, color=INK2)  # fmt: skip
    ax.set_yscale("log")
    ax.set_yticks([0.003, 0.005, 0.01, 0.02, 0.03], ["0.003", "0.005", "0.01", "0.02", "0.03"])
    ax.minorticks_off()
    ax.set_xlim(-1.5, 25)
    ax.set_xlabel("decode step time saved at 32 users (%, random weights)")
    ax.set_ylabel("verify-step KLD (log scale)")
    ax.set_title("Fast presets as first built: speed against KL divergence (LoFi experts unless noted)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "presets_tradeoff.png")


# --------------------------------------------------------------------------------------------------------------------
# 18. Serving: from the decode-only number to the served throughput (factor by factor)
# --------------------------------------------------------------------------------------------------------------------
def serving_decomp():
    factors = ["acceptance on the\nserved text", "device step in\nserving is longer", "empty seats",
               "exposed host time", "prefill calls"]  # fmt: skip
    runs = [  # title, fixed-point decode tok/s, factors in the order above, measured served tok/s
        ("(a) Before the run-ahead step", 3093, [0.846, 0.942, 0.931, 0.813, 0.805], 1503.6),
        ("(b) With the run-ahead step", 3384, [0.875, 0.888, 0.932, 0.943, 0.774], 1789.0),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.6), sharey=True)
    for ax, (title, start, fs, measured) in zip(axes, runs):
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        labels = ["decode-only fixed point"] + factors + ["served (measured)"]
        n = len(labels)
        ax.barh(n - 1, start, height=0.6, color=NEUTRAL_LIGHT, linewidth=0)
        ax.text(start + 40, n - 1, fmt_int(start), va="center", fontsize=8.5, color=INK2)
        cur = start
        for i, f in enumerate(fs):
            new = cur * f
            y = n - 2 - i
            ax.barh(y, cur - new, left=new, height=0.6, color=CAT[1], linewidth=0)
            ax.text(cur + 40, y, f"x {f:.3f}", va="center", fontsize=8.5, color=INK2)
            cur = new
        ax.barh(0, measured, height=0.6, color=CAT[0], linewidth=0)
        ax.text(measured + 40, 0, fmt_int(measured), va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(n), list(reversed(labels)), fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 3900)
        ax.set_xticks([0, 1000, 2000, 3000], ["0", "1,000", "2,000", "3,000"])
        ax.set_xlabel("output tok/s")
        ax.set_title(title, fontsize=10)
    fig.suptitle("Serving wikitext (32 concurrent, 1K in, 1K out): where the decode-only throughput goes",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")  # fmt: skip
    fig.tight_layout(w_pad=3)
    save(fig, "serving_decomp.png")


if __name__ == "__main__":
    headline()
    decode_timeline()
    custom_ops_waterfall()
    expert_op()
    plain_step_parts()
    spec_k()
    verify_parts()
    step_gains()
    kld_history()
    prefill_timeline()
    prefill_parts()
    prefill_gains()
    long_context()
    serving_timeline()
    serving_gains()
    startup()
    presets_tradeoff()
    serving_decomp()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
