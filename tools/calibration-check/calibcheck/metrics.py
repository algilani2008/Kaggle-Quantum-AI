"""Calibration metrics for probabilistic classifiers.

All functions accept:
    y_true : array of shape (n,) with integer class labels 0..K-1
    probs  : array of shape (n,) (binary: probability of class 1) or (n, K)

Conventions
-----------
* "Top-label" (confidence) calibration is used for ECE/MCE: the predicted
  class is argmax(probs) and the confidence is max(probs). For binary
  inputs given as a 1-D array this is the usual top-label definition too.
* Brier score is the multi-class form: mean over samples of
  sum_k (p_k - 1[y=k])^2. For binary 1-D input it reduces to the classic
  mean (p - y)^2 (we report that form for binary inputs).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

EPS = 1e-12


def _as_2d(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    if probs.ndim != 2:
        raise ValueError("probs must be 1-D (binary) or 2-D (n, K)")
    if np.any(probs < -1e-9) or np.any(probs > 1 + 1e-9):
        raise ValueError("probabilities must be within [0, 1]")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError("each row of probs must sum to 1")
    return np.clip(probs, 0.0, 1.0)


def _check_labels(y_true, n_classes: int) -> np.ndarray:
    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError("y_true must be 1-D")
    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(np.equal(np.mod(y, 1), 0)):
            raise ValueError("y_true must contain integer class labels")
        y = y.astype(int)
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError(f"labels must be in [0, {n_classes - 1}]")
    return y


def _bin_edges(conf: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ValueError("strategy must be 'uniform' or 'quantile'")


@dataclass
class ReliabilityBins:
    """Per-bin statistics behind a reliability diagram."""

    edges: np.ndarray
    counts: np.ndarray
    mean_confidence: np.ndarray  # NaN for empty bins
    accuracy: np.ndarray  # NaN for empty bins

    def to_rows(self):
        rows = []
        for i in range(len(self.counts)):
            rows.append(
                {
                    "bin_low": float(self.edges[i]),
                    "bin_high": float(self.edges[i + 1]),
                    "count": int(self.counts[i]),
                    "mean_confidence": float(self.mean_confidence[i]),
                    "accuracy": float(self.accuracy[i]),
                }
            )
        return rows


def reliability_bins(y_true, probs, n_bins: int = 15, strategy: str = "uniform") -> ReliabilityBins:
    """Group samples by top-label confidence and compute accuracy per bin."""
    p = _as_2d(probs)
    y = _check_labels(y_true, p.shape[1])
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = _bin_edges(conf, n_bins, strategy)
    # bins are (low, high]; the first bin also includes 0
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    counts = np.bincount(idx, minlength=nb)
    sum_conf = np.bincount(idx, weights=conf, minlength=nb)
    sum_acc = np.bincount(idx, weights=correct, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(counts > 0, sum_conf / np.maximum(counts, 1), np.nan)
        acc = np.where(counts > 0, sum_acc / np.maximum(counts, 1), np.nan)
    return ReliabilityBins(edges, counts, mean_conf, acc)


def expected_calibration_error(y_true, probs, n_bins: int = 15, strategy: str = "uniform") -> float:
    """ECE = sum_b (n_b / n) * |acc_b - conf_b| over non-empty bins."""
    b = reliability_bins(y_true, probs, n_bins, strategy)
    mask = b.counts > 0
    w = b.counts[mask] / b.counts.sum()
    return float(np.sum(w * np.abs(b.accuracy[mask] - b.mean_confidence[mask])))


def maximum_calibration_error(y_true, probs, n_bins: int = 15, strategy: str = "uniform") -> float:
    """MCE = max over non-empty bins of |acc_b - conf_b|."""
    b = reliability_bins(y_true, probs, n_bins, strategy)
    mask = b.counts > 0
    return float(np.max(np.abs(b.accuracy[mask] - b.mean_confidence[mask])))


def brier_score(y_true, probs) -> float:
    """Brier score. Binary 1-D input -> mean (p - y)^2; 2-D -> multi-class sum form."""
    arr = np.asarray(probs, dtype=float)
    p = _as_2d(arr)
    y = _check_labels(y_true, p.shape[1])
    if arr.ndim == 1:
        return float(np.mean((p[:, 1] - y) ** 2))
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def log_loss(y_true, probs) -> float:
    """Negative log-likelihood (natural log), probabilities clipped at 1e-12."""
    p = _as_2d(probs)
    y = _check_labels(y_true, p.shape[1])
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))))


def accuracy(y_true, probs) -> float:
    p = _as_2d(probs)
    y = _check_labels(y_true, p.shape[1])
    return float(np.mean(p.argmax(axis=1) == y))


def brier_decomposition(y_true, probs, n_bins: int = 10) -> dict:
    """Murphy decomposition of the binary Brier score: BS ~= REL - RES + UNC.

    Only defined here for binary problems (1-D probs or 2 columns). Forecasts
    are grouped into ``n_bins`` equal-width bins of p(class 1), so the identity
    holds exactly only when forecasts within a bin are equal; the residual is
    reported as ``within_bin`` so REL - RES + UNC + within_bin == BS.
    """
    p = _as_2d(probs)
    if p.shape[1] != 2:
        raise ValueError("brier_decomposition supports binary problems only")
    y = _check_labels(y_true, 2).astype(float)
    f = p[:, 1]
    n = len(y)
    base = y.mean()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, f, side="left") - 1, 0, n_bins - 1)
    rel = res = 0.0
    for k in range(n_bins):
        m = idx == k
        nk = m.sum()
        if nk == 0:
            continue
        fk, ok = f[m].mean(), y[m].mean()
        rel += nk * (fk - ok) ** 2
        res += nk * (ok - base) ** 2
    rel, res = rel / n, res / n
    unc = base * (1 - base)
    bs = float(np.mean((f - y) ** 2))
    return {
        "brier": bs,
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "within_bin": float(bs - (rel - res + unc)),
    }


@dataclass
class CalibrationReport:
    n_samples: int
    n_classes: int
    accuracy: float
    ece: float
    ece_quantile: float
    mce: float
    brier: float
    log_loss: float
    mean_confidence: float
    n_bins: int

    def to_dict(self):
        return asdict(self)


def calibration_report(y_true, probs, n_bins: int = 15) -> CalibrationReport:
    """Compute the standard set of calibration metrics in one call."""
    p = _as_2d(probs)
    return CalibrationReport(
        n_samples=int(p.shape[0]),
        n_classes=int(p.shape[1]),
        accuracy=accuracy(y_true, probs),
        ece=expected_calibration_error(y_true, probs, n_bins, "uniform"),
        ece_quantile=expected_calibration_error(y_true, probs, n_bins, "quantile"),
        mce=maximum_calibration_error(y_true, probs, n_bins, "uniform"),
        brier=brier_score(y_true, probs),
        log_loss=log_loss(y_true, probs),
        mean_confidence=float(p.max(axis=1).mean()),
        n_bins=n_bins,
    )
