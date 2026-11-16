# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The figures of PERF.md, from the measured values written below (the same as PERF.md's tables).

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
    total = [1089, 1795, 3335, 4983, 7342]
    per_user = [136, 112, 104, 78, 57]
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
        "128": [246, 647, 948, 1763, 3273],
        "1K": [127, 392, 797, 1048, 1846],
        "8K": [116, 326, 500, 736, 1037],
        "32K": [126, 283, 357, 316, 359],
        "128K": [82, 106, 103, None, None],
    }
    ttft = {
        "128": [0.17, 0.24, 0.23, 0.26, 0.46],
        "1K": [0.17, 0.30, 0.32, 0.43, 0.57],
        "8K": [0.52, 0.56, 0.57, 0.80, 1.55],
        "32K": [2.3, 4.9, 9.0, 17.4, 34.4],
        "128K": [8.3, 20.4, 38.6, None, None],
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
    b.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20, 50], ["0.2", "0.5", "1", "2", "5", "10", "20", "50"])
    b.minorticks_off()
    a.set_ylim(0, 3600)
    a.legend(loc="upper left", fontsize=8.5, title="input tokens", title_fontsize=8.5)
    fig.tight_layout(w_pad=3)
    save(fig, "serving.png")


def presets():
    users = ["8 users", "32 users", "128 users"]
    gain = {
        "A (F = 0.7)": [8.2, 8.3, 5.4],
        "B (F = 0.6)": [13.1, 14.1, 9.7],
        "C (F = 0.5)": [18.7, 21.0, 15.1],
        "all BFP4": [19.5, 22.3, 15.1],
    }
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.19
    for i, (name, vals) in enumerate(gain.items()):
        xs = [k + (i - 1.5) * w for k in range(len(users))]
        bars = ax.bar(xs, vals, width=w, color=CAT[i], edgecolor=SURFACE, linewidth=1.5, label=name)
        bar_labels(ax, bars, "+{:.0f}%")
    ax.set_xticks(range(len(users)), users)
    ax.set_ylabel("decode tok/s against the default (%)")
    ax.set_ylim(0, 26)
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


if __name__ == "__main__":
    decode()
    serving()
    presets()
    quality()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
