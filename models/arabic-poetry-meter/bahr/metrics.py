"""Small, dependency-light classification metrics."""

from __future__ import annotations

import numpy as np


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("empty input")
    return float(np.mean(y_true == y_pred))


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Rows = true label, columns = predicted label."""
    m = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(m, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return m


def per_class_report(y_true, y_pred, n_classes: int) -> list[dict]:
    m = confusion(y_true, y_pred, n_classes)
    rows = []
    for k in range(n_classes):
        tp = m[k, k]
        support = int(m[k].sum())
        predicted = int(m[:, k].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": k, "support": support, "precision": float(precision), "recall": float(recall), "f1": float(f1)})
    return rows


def macro_f1(y_true, y_pred) -> float:
    """Mean F1 over the classes that occur in y_true or y_pred."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.union1d(y_true, y_pred)
    remap = {v: i for i, v in enumerate(labels)}
    t = np.array([remap[v] for v in y_true])
    p = np.array([remap[v] for v in y_pred])
    rows = per_class_report(t, p, len(labels))
    return float(np.mean([r["f1"] for r in rows]))


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for accuracy (resampling verses)."""
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(accs, a)), float(np.quantile(accs, 1 - a))
