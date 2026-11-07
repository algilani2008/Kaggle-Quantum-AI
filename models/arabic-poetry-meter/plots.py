"""Draw the result charts from results/*.json and *.csv (run after train.py)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
INK, MUTED, ACCENT, GRID = "#1f2933", "#8a94a6", "#2563eb", "#e5e7eb"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ablation_chart(metrics):
    rows = [r for r in metrics["validation"] if not r["model"].startswith("full") or f"C={metrics['chosen_C']}" in r["model"]]
    names = [r["model"].replace(f", C={metrics['chosen_C']}", "").replace(", + ", " + ") for r in rows]
    vals = [r["val_accuracy"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    colors = [ACCENT if n.startswith("full") else "#9db8ef" for n in names]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 105)
    ax.set_xlabel("validation accuracy (%)", color=INK)
    ax.set_title("What each feature group adds (validation set, 4,713 verses)", loc="left", color=INK, fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(RES / "ablation.png", dpi=150)
    plt.close(fig)


def per_meter_chart():
    with open(RES / "per_meter.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: float(r["f1"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = np.arange(len(rows))
    f1 = [float(r["f1"]) * 100 for r in rows]
    ax.barh(y, f1, color=ACCENT, height=0.6)
    for yi, v, r in zip(y, f1, rows):
        ax.text(v + 1, yi, f"{v:.0f}  (n={r['support']})", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [r["meter"] for r in rows])
    ax.set_xlim(0, 112)
    ax.set_xlabel("F1 on the official test split (%)", color=INK)
    ax.set_title("Per-meter F1 (test, 8,316 verses)", loc="left", color=INK, fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(RES / "per_meter_f1.png", dpi=150)
    plt.close(fig)


def confusion_chart():
    with open(RES / "confusion_matrix.csv", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    m = np.loadtxt(RES / "confusion_matrix.csv", delimiter=",", skiprows=1)
    norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(header)), header, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(header)), header, fontsize=8)
    for i in range(len(header)):
        for j in range(len(header)):
            if norm[i, j] >= 0.03:
                ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xlabel("predicted meter")
    ax.set_ylabel("true meter")
    ax.set_title("Confusion matrix, % of each true meter (test)", loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(RES / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def diacritics_chart(metrics):
    rows = metrics["final_model_test"]["by_diacritization"]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    y = np.arange(len(rows))[::-1]
    vals = [r["accuracy"] * 100 for r in rows]
    ax.barh(y, vals, color=ACCENT, height=0.55)
    for yi, v, r in zip(y, vals, rows):
        ax.text(v + 1, yi, f"{v:.1f}%  (n={r['n']})", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r["diacritics"] for r in rows])
    ax.set_xlim(0, 112)
    ax.set_xlabel("test accuracy (%)")
    ax.set_title("Accuracy by how much of the verse carries diacritics", loc="left", fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(RES / "accuracy_by_diacritics.png", dpi=150)
    plt.close(fig)


def main():
    metrics = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))
    ablation_chart(metrics)
    per_meter_chart()
    confusion_chart()
    diacritics_chart(metrics)
    print("charts written to", RES)


if __name__ == "__main__":
    main()
