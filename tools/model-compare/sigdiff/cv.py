"""Tests for comparing two models from cross-validation fold scores.

Fold scores are *not* independent: training sets overlap heavily and, with
repeated CV, every row is tested many times. A plain paired t-test on fold
scores therefore underestimates the variance. Two standard fixes:

* :func:`corrected_resampled_ttest` (Nadeau & Bengio, 2003): inflate the
  variance by ``1/J + n_test/n_train``.
* :func:`five_by_two_cv_ttest` (Dietterich, 1998): 5 repetitions of 2-fold CV.

:func:`naive_ttest` is included only so you can see the difference.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .paired import TestResult


def _diffs(scores_a, scores_b) -> np.ndarray:
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b need the same number of folds")
    if a.size < 2:
        raise ValueError("need at least two folds")
    return b - a


def _t_result(method, mean, se, df, alpha):
    if se == 0:
        p = 1.0 if mean == 0 else 0.0
        return TestResult(method, float(mean), p, float(mean), float(mean))
    t = mean / se
    p = 2 * stats.t.sf(abs(t), df)
    q = stats.t.ppf(1 - alpha / 2, df)
    return TestResult(method, float(mean), float(p), float(mean - q * se), float(mean + q * se))


def naive_ttest(scores_a, scores_b, alpha=0.05) -> TestResult:
    """Ordinary paired t-test on fold scores. Usually over-confident; for reference only."""
    d = _diffs(scores_a, scores_b)
    j = d.size
    return _t_result("naive_ttest", d.mean(), d.std(ddof=1) / np.sqrt(j), j - 1, alpha)


def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test, alpha=0.05) -> TestResult:
    """Nadeau-Bengio corrected resampled t-test.

    ``scores_*`` are per-fold scores from the same folds (e.g. 10x10 repeated
    k-fold). ``n_train`` / ``n_test`` are the sizes of one training / test fold.
    """
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive")
    d = _diffs(scores_a, scores_b)
    j = d.size
    var = d.var(ddof=1) * (1.0 / j + n_test / n_train)
    return _t_result("corrected_resampled_ttest", d.mean(), np.sqrt(var), j - 1, alpha)


def five_by_two_cv_ttest(scores_a, scores_b, alpha=0.05) -> TestResult:
    """Dietterich's 5x2cv paired t-test.

    ``scores_*`` must have shape (5, 2): 5 repetitions x 2 folds, with fold 0
    and fold 1 of each repetition being the two halves of one random split.
    The reported ``diff`` is the mean over all 10 folds; the t statistic uses
    the first fold's difference, as in the original paper.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != (5, 2) or b.shape != (5, 2):
        raise ValueError("5x2cv needs score arrays of shape (5, 2)")
    d = b - a
    mean_i = d.mean(axis=1)
    s2 = ((d - mean_i[:, None]) ** 2).sum(axis=1)
    denom = np.sqrt(s2.mean())
    if denom == 0:
        return TestResult("five_by_two_cv_ttest", float(d.mean()), 1.0 if d[0, 0] == 0 else 0.0)
    t = d[0, 0] / denom
    p = 2 * stats.t.sf(abs(t), 5)
    # Heuristic interval: mean of the 10 fold differences +/- t * (the test's
    # standard error). Conservative (wide); the p-value is the standard 5x2cv one.
    q = stats.t.ppf(1 - alpha / 2, 5)
    m = d.mean()
    return TestResult("five_by_two_cv_ttest", float(m), float(p), float(m - q * denom), float(m + q * denom))
