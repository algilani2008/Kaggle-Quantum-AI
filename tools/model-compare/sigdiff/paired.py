"""Paired tests for two models scored on the *same* test rows.

All functions return the difference ``metric(B) - metric(A)`` in the metric's
own units. Use :func:`sigdiff.compare` for a friendlier wrapper that also turns
this into "which model is better".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import binomtest

from .metrics import Metric, _as_labels, get_metric


@dataclass
class TestResult:
    method: str
    diff: float                 # metric(B) - metric(A) on the full test set
    p_value: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    n_resamples: int = 0


def _check(y, a, b) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    if not (y.shape[0] == a.shape[0] == b.shape[0]):
        raise ValueError("y, pred_a and pred_b must have the same length")
    if y.shape[0] < 2:
        raise ValueError("need at least two test rows")
    return y, a, b


def _group_index(groups, n):
    """Return (list of row-index arrays per group). Each row is its own group if groups is None."""
    if groups is None:
        return None
    groups = np.asarray(groups)
    if groups.shape[0] != n:
        raise ValueError("groups must have one entry per test row")
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    return np.split(order, np.cumsum(counts)[:-1])


def paired_bootstrap(y, pred_a, pred_b, metric="accuracy", *, n_resamples=2000,
                     alpha=0.05, groups=None, seed: Optional[int] = 0) -> TestResult:
    """Paired (optionally cluster) bootstrap of ``metric(B) - metric(A)``.

    The same resampled rows are used for both models, so row-to-row difficulty
    cancels out. Returns a percentile confidence interval and a two-sided
    bootstrap p-value ``2 * min(P(d* <= 0), P(d* >= 0))``.
    """
    m: Metric = get_metric(metric)
    y, a, b = _check(y, pred_a, pred_b)
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    observed = m.score(y, b) - m.score(y, a)
    blocks = _group_index(groups, n)

    diffs = np.empty(n_resamples)
    if blocks is None and m.decomposable:
        d = m.pointwise(y, b) - m.pointwise(y, a)
        chunk = max(1, int(2_000_000 // n))
        for start in range(0, n_resamples, chunk):
            stop = min(n_resamples, start + chunk)
            idx = rng.integers(0, n, size=(stop - start, n))
            diffs[start:stop] = d[idx].mean(axis=1)
    else:
        for r in range(n_resamples):
            if blocks is None:
                idx = rng.integers(0, n, size=n)
            else:
                pick = rng.integers(0, len(blocks), size=len(blocks))
                idx = np.concatenate([blocks[k] for k in pick])
            diffs[r] = m.score(y[idx], b[idx]) - m.score(y[idx], a[idx])

    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return TestResult("paired_bootstrap", float(observed), float(min(1.0, p)), float(lo), float(hi), int(diffs.size))


def permutation_test(y, pred_a, pred_b, metric="accuracy", *, n_resamples=2000,
                     groups=None, seed: Optional[int] = 0) -> TestResult:
    """Paired permutation (swap) test.

    Under the null hypothesis "A and B are interchangeable", swapping the two
    models' predictions on any row (or any group of rows) does not change the
    distribution of the score difference. We swap at random and count how often
    the swapped difference is at least as extreme as the observed one.
    Works for any metric, including AUC and F1.
    """
    m: Metric = get_metric(metric)
    y, a, b = _check(y, pred_a, pred_b)
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    observed = m.score(y, b) - m.score(y, a)
    blocks = _group_index(groups, n)

    null = np.empty(n_resamples)
    if blocks is None and m.decomposable:
        d = m.pointwise(y, b) - m.pointwise(y, a)
        chunk = max(1, int(2_000_000 // n))
        for start in range(0, n_resamples, chunk):
            stop = min(n_resamples, start + chunk)
            signs = rng.choice((-1.0, 1.0), size=(stop - start, n))
            null[start:stop] = (signs * d).mean(axis=1)
    else:
        for r in range(n_resamples):
            if blocks is None:
                swap = rng.random(n) < 0.5
            else:
                gswap = rng.random(len(blocks)) < 0.5
                swap = np.zeros(n, dtype=bool)
                for k in np.flatnonzero(gswap):
                    swap[blocks[k]] = True
            aa = np.where(swap, b, a)
            bb = np.where(swap, a, b)
            null[r] = m.score(y, bb) - m.score(y, aa)

    tol = 1e-12 * max(1.0, abs(observed))
    p = (1 + np.sum(np.abs(null) >= abs(observed) - tol)) / (1 + n_resamples)
    return TestResult("permutation", float(observed), float(p), n_resamples=n_resamples)


def mcnemar(y, pred_a, pred_b) -> TestResult:
    """Exact McNemar test for two classifiers' accuracy on the same rows.

    Only rows where exactly one model is right carry information; under the
    null each such row is a fair coin flip.
    """
    y, a, b = _check(y, pred_a, pred_b)
    ca = np.asarray(y) == _as_labels(a)
    cb = np.asarray(y) == _as_labels(b)
    only_b = int(np.sum(cb & ~ca))
    only_a = int(np.sum(ca & ~cb))
    diff = (only_b - only_a) / y.shape[0]
    if only_a + only_b == 0:
        return TestResult("mcnemar", 0.0, 1.0)
    p = binomtest(only_b, only_a + only_b, 0.5).pvalue
    return TestResult("mcnemar", float(diff), float(p))
