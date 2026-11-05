"""Classification metrics, calibration error and a book-level bootstrap."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, prob, bins: int = 10) -> float:
    """Weighted gap between stated probability and observed positive rate."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    index = np.minimum((prob * bins).astype(int), bins - 1)
    error = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - prob[mask].mean())
    return float(error)


def summarize(y_true, prob, threshold: float = 0.5) -> dict[str, float]:
    """Return the headline metrics for positive-class probabilities."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    pred = (prob >= threshold).astype(int)
    result = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "f1_negative": f1_score(y_true, pred, pos_label=0, zero_division=0),
        "f1_positive": f1_score(y_true, pred, pos_label=1, zero_division=0),
    }
    if len(np.unique(prob)) > 1:
        result["roc_auc"] = roc_auc_score(y_true, prob)
    result["brier"] = brier_score_loss(y_true, prob)
    result["ece"] = expected_calibration_error(y_true, prob)
    return {k: float(v) for k, v in result.items()}


def best_threshold(y_true, prob) -> float:
    """Pick the probability cut with the best macro-F1 on validation data."""
    grid = np.round(np.arange(0.10, 0.91, 0.02), 2)
    scores = [
        f1_score(y_true, (np.asarray(prob) >= t).astype(int), average="macro", zero_division=0)
        for t in grid
    ]
    return float(grid[int(np.argmax(scores))])


def _book_counts(y_true, pred, groups) -> np.ndarray:
    """Return TP, FP, FN, TN for the positive class, one row per book."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    _, book = np.unique(np.asarray(groups), return_inverse=True)
    counts = np.zeros((book.max() + 1, 4))
    np.add.at(counts[:, 0], book, (y_true == 1) & (pred == 1))
    np.add.at(counts[:, 1], book, (y_true == 0) & (pred == 1))
    np.add.at(counts[:, 2], book, (y_true == 1) & (pred == 0))
    np.add.at(counts[:, 3], book, (y_true == 0) & (pred == 0))
    return counts


def _macro_f1(counts: np.ndarray) -> np.ndarray:
    """Macro-F1 from arrays of TP, FP, FN, TN stored in the last axis."""
    tp, fp, fn, tn = np.moveaxis(counts, -1, 0)
    pos = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    neg = 2 * tn / np.maximum(2 * tn + fp + fn, 1)
    return (pos + neg) / 2


def book_bootstrap(y_true, pred_a, groups, pred_b=None, reps: int = 2000, seed: int = 0) -> dict:
    """95% interval for macro-F1 that resamples whole books.

    Reviews of one book are not independent, so resampling single reviews
    would make the interval too narrow. With ``pred_b`` the interval is for
    the paired difference macro-F1(a) - macro-F1(b).
    """
    rng = np.random.default_rng(seed)
    counts_a = _book_counts(y_true, pred_a, groups)
    n = len(counts_a)
    draws = rng.multinomial(n, np.full(n, 1 / n), size=reps)
    stats = _macro_f1(draws @ counts_a)
    point = float(_macro_f1(counts_a.sum(0)))
    if pred_b is not None:
        counts_b = _book_counts(y_true, pred_b, groups)
        stats = stats - _macro_f1(draws @ counts_b)
        point -= float(_macro_f1(counts_b.sum(0)))
    low, high = np.percentile(stats, [2.5, 97.5])
    return {"estimate": point, "low": float(low), "high": float(high)}
