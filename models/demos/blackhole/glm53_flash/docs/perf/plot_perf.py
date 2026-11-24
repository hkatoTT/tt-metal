# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The figures of PERF.md and README.md (progressive.png: a toy example), from the measured values written below (the same as PERF.md's tables).

    python models/demos/blackhole/glm53_flash/docs/perf/plot_perf.py   # needs matplotlib; writes the PNGs beside it
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent

# chart surface and ink
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# categorical slots, in this fixed order (validated for color-vision deficiencies on adjacent series)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# an ordinal blue ramp, light to dark (validated: monotone, the lightest step still 2:1 on the surface)
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

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


def bar_labels(ax, bars, fmt="{:,.0f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK2)  # fmt: skip


def decode():
    users = [8, 16, 32, 64, 128]
    total = [1208, 1881, 3754, 5718, 8469]
    per_user = [151, 118, 117, 89, 66]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    x = range(len(users))
    for ax, vals, title in ((a, total, "Decode: total tok/s"), (b, per_user, "Decode: tok/s per user")):
        bars = ax.bar(x, vals, width=0.6, color=CAT[0], edgecolor=SURFACE, linewidth=1)
        bar_labels(ax, bars)
        ax.set_xticks(list(x), [str(u) for u in users])
        ax.set_xlabel("users (7 drafts a step up to 64 users, 3 at 128)")
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout(w_pad=3)
    save(fig, "decode.png")


def serving():
    conc = [1, 4, 8, 16, 32]
    tput = {
        "128": [283, 801, 1038, 2131, 3300],
        "1K": [257, 449, 694, 1329, 2450],
        "8K": [188, 313, 847, 871, 1168],
        "32K": [147, 378, 538, 607, 562],
        "128K": [91, 188, 197, None, None],
    }
    ttft = {
        "128": [0.08, 0.20, 0.23, 0.25, 0.40],
        "1K": [0.12, 0.33, 0.48, 0.50, 0.89],
        "8K": [0.42, 0.47, 0.49, 0.93, 1.30],
        "32K": [1.70, 3.19, 4.14, 6.70, 13.95],
        "128K": [7.25, 11.73, 16.52, None, None],
    }
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, data, title, nudge in (
        (a, tput, "Serving: output tok/s", {}),
        (b, ttft, "Serving: time to first token (s, median, log scale)", {"1K": 4, "128": -4}),
    ):
        for i, (name, vals) in enumerate(data.items()):
            pts = [(c, v) for c, v in zip(conc, vals) if v is not None]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=ORD[i], linewidth=1.8, marker="o", markersize=5, label=f"input {name}")
            ax.annotate(f"{name}", (xs[-1], ys[-1]), xytext=(6, nudge.get(name, 0)), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK2)  # fmt: skip
        ax.set_xscale("log", base=2)
        ax.set_xticks(conc, [str(c) for c in conc])
        ax.minorticks_off()
        ax.set_xlabel("concurrent requests (32 seats, 1,024 output tokens)")
        ax.set_title(title)
        ax.set_xlim(0.8, 48)
    b.set_yscale("log")
    b.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20], ["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
    b.minorticks_off()
    a.set_ylim(0, 3600)
    a.legend(loc="upper left", fontsize=8.5, title="input tokens", title_fontsize=8.5)
    fig.tight_layout(w_pad=3)
    save(fig, "serving.png")


def presets():
    users = ["8 users", "32 users", "64 users", "128 users"]
    gain = {
        "A (F = 0.7)": [8.0, 9.5, 8.5, 7.5],
        "B (F = 0.6)": [12.5, 14.9, 11.2, 10.9],
        "C (F = 0.5)": [19.0, 22.6, 18.1, 19.0],
        "all BFP4": [15.8, 17.4, 14.1, 12.8],
    }
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    w = 0.19
    for i, (name, vals) in enumerate(gain.items()):
        xs = [k + (i - 1.5) * w for k in range(len(users))]
        bars = ax.bar(xs, vals, width=w, color=CAT[i], edgecolor=SURFACE, linewidth=1.5, label=name)
        bar_labels(ax, bars, "+{:.0f}%")
    ax.set_xticks(range(len(users)), users)
    ax.set_ylabel("decode tok/s against the default (%)")
    ax.set_ylim(0, 28)
    ax.set_title("Fast presets: decode speed gain over the default")
    ax.legend(ncol=4, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "presets.png")


def quality():
    caps = ["32K", "64K", "128K"]
    score = {
        "MATH-500": [98.0, 98.0, 98.0],
        "IFEval": [92.0, 92.3, 92.3],
        "GPQA Diamond": [87.8, 92.8, 95.0],
        "AIME 2025": [70.0, 84.4, 87.8],
        "AIME 2024": [77.8, 84.4, 88.9],
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, (name, vals) in enumerate(score.items()):
        ax.plot(range(len(caps)), vals, color=CAT[i], linewidth=1.8, marker="o", markersize=5, label=name)
        nudge = {"AIME 2024": 3, "AIME 2025": -4}.get(name, 0)  # (points: the two AIME end labels 1.1 apart)
        ax.annotate(f"{name} {vals[-1]:.0f}%", (len(caps) - 1, vals[-1]), xytext=(7, nudge), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)  # fmt: skip
    ax.set_xticks(range(len(caps)), caps)
    ax.set_xlim(-0.2, 2.9)
    ax.set_ylim(65, 101)
    ax.set_xlabel("cap on generated tokens (AIME 2024: 96K at the last point)")
    ax.set_ylabel("solved within the cap (%)")
    ax.set_title("Task quality: temperature 1.0, top-p 0.95, reasoning max, mean of 3 seeds")
    fig.tight_layout()
    save(fig, "quality.png")


def progressive():
    """The fast presets' read budget, a toy step: 4 mesh columns choosing 10 / 8 / 7 / 5 experts (sizes rounded: one
    expert in BFP8 = 1, its BFP4 base and its BFP4 correction 0.5 each; the real base is about 0.53)."""
    chosen = [10, 8, 7, 5]  # experts chosen in this step by each mesh column (column 0 the busiest)
    full = "#898781"  # one expert's whole BFP8 weight (the default)
    base, corr = CAT[0], CAT[1]
    forms = [
        ("Default: every chosen expert in BFP8", None),
        ("Preset A (F = 0.7): budget = 0.7 x the busiest column's BFP8 bytes = 7", 0.7),
        ("Preset C (F = 0.5): budget = 5; the busiest column reads no correction", 0.5),
        ("All-BFP4 (comparison): bases only", 0.0),
    ]
    fig, axes = plt.subplots(len(forms), 1, figsize=(8.2, 7.4), sharex=True)
    gap = 0.04
    for ax, (title, f) in zip(axes, forms):
        ax.grid(False)
        for row, n in enumerate(chosen):
            y = len(chosen) - 1 - row
            x = 0.0
            if f is None:
                for _ in range(n):
                    ax.barh(y, 1.0 - gap, left=x, height=0.62, color=full, linewidth=0)
                    x += 1.0
            else:
                budget = f * max(chosen)
                m = 0 if f == 0.0 else max(0, min(n, int(round((budget - 0.5 * n) / 0.5, 6))))
                for e in range(n):  # left: the largest routing-weight sum in this column
                    if e < m:  # one expert: its base and its correction side by side, a hairline between
                        ax.barh(y, 0.5 - 0.012, left=x, height=0.62, color=base, linewidth=0)
                        ax.barh(y, 0.5 - gap, left=x + 0.5, height=0.62, color=corr, linewidth=0)
                        x += 1.0
                    else:
                        ax.barh(y, 0.5 - gap, left=x, height=0.62, color=base, linewidth=0)
                        x += 0.5
            ax.text(x + 0.12, y, f"{x:g}", va="center", fontsize=8.5, color=INK2)
        if f:
            ax.axvline(f * max(chosen), color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.text(f * max(chosen), len(chosen) - 0.45, " budget, the same for every column", fontsize=8, color=INK,
                    va="bottom")  # fmt: skip
        ax.set_yticks(range(len(chosen)), [f"column {c}: {n} experts" for c, n in reversed(list(enumerate(chosen)))])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(chosen) - 0.1)
        ax.set_title(title, fontsize=10)
    axes[-1].set_xlim(0, 11)
    axes[-1].set_xlabel("weight bytes read by each mesh column in one decode step (one expert in BFP8 = 1)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=full),
        plt.Rectangle((0, 0), 1, 1, color=base),
        plt.Rectangle((0, 0), 1, 1, color=corr),
    ]
    fig.legend(handles, ["expert in BFP8 (default)", "BFP4 base", "BFP4 correction"], ncol=3, loc="upper left",
               bbox_to_anchor=(0.01, 1.045), fontsize=8.5)  # fmt: skip
    fig.text(0.01, 0.995, "Each block is one chosen expert, the largest sum of routing weights in this step on the left; "
             "A and C add the correction\nto the leftmost experts while the column stays within the budget.",
             ha="left", va="top", fontsize=8.5, color=INK2)  # fmt: skip
    fig.suptitle("How F sets the experts read with their correction (a toy step; the step waits for the longest bar)",
                 x=0.01, y=1.085, ha="left", fontsize=11, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    save(fig, "progressive.png")


if __name__ == "__main__":
    decode()
    serving()
    presets()
    quality()
    progressive()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
