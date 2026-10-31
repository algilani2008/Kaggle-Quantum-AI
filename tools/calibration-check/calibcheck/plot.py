"""Reliability diagram (matplotlib, headless-safe)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import reliability_bins, expected_calibration_error  # noqa: E402


def reliability_diagram(curves, path, n_bins: int = 15, title: str = "Reliability diagram"):
    """Draw one or more reliability curves and save to ``path``.

    curves: list of (label, y_true, probs) tuples.
    """
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(6, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="perfect")
    colors = ["#2a6fdb", "#e0642b", "#1b9e77", "#7b3fb5", "#c23b6b"]
    for i, (label, y, p) in enumerate(curves):
        b = reliability_bins(y, p, n_bins)
        m = b.counts > 0
        ece = expected_calibration_error(y, p, n_bins)
        c = colors[i % len(colors)]
        ax.plot(b.mean_confidence[m], b.accuracy[m], marker="o", markersize=4,
                color=c, linewidth=1.8, label=f"{label} (ECE {ece:.3f})")
        centers = (b.edges[:-1] + b.edges[1:]) / 2
        width = (b.edges[1:] - b.edges[:-1]) / (len(curves) + 0.5)
        axh.bar(centers + (i - (len(curves) - 1) / 2) * width, b.counts, width=width,
                color=c, alpha=0.8)
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    axh.set_xlabel("predicted confidence (top label)")
    axh.set_ylabel("count")
    axh.set_yscale("log")
    axh.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
