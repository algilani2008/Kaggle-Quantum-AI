"""Charts for results/. One series each, so the title names it and no legend is needed."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _axes(size):
    fig, ax = plt.subplots(figsize=size, dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def comparison_chart(summary: dict, path: Path) -> None:
    ci = summary["intervals_95_book_bootstrap"]
    rows = [("Always say majority", ci["majority_macro_f1"]),
            ("Naive Bayes (baseline)", ci["naive_bayes_macro_f1"]),
            (f"Final model ({summary['selected']['features']})", ci["final_macro_f1"])]
    fig, ax = _axes((7, 2.6))
    y = np.arange(len(rows))
    values = np.array([r[1]["estimate"] for r in rows])
    low = values - np.array([r[1]["low"] for r in rows])
    high = np.array([r[1]["high"] for r in rows]) - values
    ax.barh(y, values, height=0.28, color=BLUE)
    ax.errorbar(values, y, xerr=[low, high], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    for i, v in enumerate(values):
        ax.text(values[i] + high[i] + 0.012, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 on unseen books (bars: 95% book bootstrap)", color=MUTED, fontsize=9)
    ax.set_title("Test results", loc="left", fontsize=11, color=INK)
    _finish(fig, path)


def leakage_chart(audit, official: dict | None, path: Path) -> None:
    groups = [("Unseen books", audit[audit.split == "book"]["macro_f1"].to_numpy()),
              ("Random split, duplicates removed", audit[audit.split == "random"]["macro_f1"].to_numpy())]
    if official:
        groups.append(("Official LABR split, as published", np.array([official["macro_f1"]])))
    fig, ax = _axes((7, 2.6))
    for i, (_, scores) in enumerate(groups):
        ax.scatter(scores, np.full(len(scores), i), s=42, color=BLUE, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.text(scores.max() + 0.003, i, f"{scores.mean():.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups], color=INK)
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.set_xlabel("Macro-F1 of the same model (dot: one seed, number: mean)", color=MUTED, fontsize=9)
    ax.set_title("How the split changes the score", loc="left", fontsize=11, color=INK)
    _finish(fig, path)


def calibration_chart(y_true, prob, path: Path, bins: int = 10) -> None:
    index = np.minimum((np.asarray(prob) * bins).astype(int), bins - 1)
    used = [b for b in range(bins) if (index == b).any()]
    stated = [np.asarray(prob)[index == b].mean() for b in used]
    observed = [np.asarray(y_true)[index == b].mean() for b in used]
    fig, ax = _axes((4.4, 4.2))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.17, 0.27, "perfect", color=MUTED, fontsize=8, rotation=42)
    ax.plot(stated, observed, color=BLUE, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probability stated by the model", color=MUTED, fontsize=9)
    ax.set_ylabel("Share truly positive", color=MUTED, fontsize=9)
    ax.set_title("Calibration on unseen books", loc="left", fontsize=11, color=INK)
    _finish(fig, path)


def length_chart(lengths, path: Path) -> None:
    fig, ax = _axes((6, 2.5))
    y = np.arange(len(lengths))
    ax.barh(y, lengths["macro_f1"], height=0.32, color=BLUE)
    for i, row in lengths.iterrows():
        ax.text(row["macro_f1"] + 0.012, i, f"{row['macro_f1']:.3f}  (n={row['reviews']:,})",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y, lengths["characters"], color=INK)
    ax.set_xlim(0, 1.15)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("Macro-F1 on unseen books", color=MUTED, fontsize=9)
    ax.set_title("Final model by review length (characters)", loc="left", fontsize=11, color=INK)
    _finish(fig, path)
