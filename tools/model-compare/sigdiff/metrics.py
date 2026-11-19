"""Metric registry used by the comparison tests.

Every metric is described by a :class:`Metric` with

* ``score(y, pred)``: the usual aggregate value on a whole test set,
* ``pointwise(y, pred)``: per-row contributions whose *mean* equals the score
  (``None`` for metrics such as AUC or F1 that do not decompose per row),
* ``greater_is_better``: the direction of improvement.

Decomposable metrics allow fast vectorised resampling; the others are handled
by recomputing the full metric on every resample.

Only numpy and scipy are needed here, so the core library stays light.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy.stats import rankdata

_EPS = 1e-15


def _as_labels(pred: np.ndarray) -> np.ndarray:
    """Turn probabilities (floats in [0, 1]) into 0/1 labels; keep labels as is."""
    pred = np.asarray(pred)
    if pred.dtype.kind == "f" and pred.size and np.all((pred >= 0) & (pred <= 1)) and not np.all(np.isin(pred, (0.0, 1.0))):
        return (pred >= 0.5).astype(int)
    return pred


def _accuracy_pointwise(y, pred):
    return (np.asarray(y) == _as_labels(pred)).astype(float)


def _logloss_pointwise(y, p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _brier_pointwise(y, p):
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def _abs_err(y, p):
    return np.abs(np.asarray(p, dtype=float) - np.asarray(y, dtype=float))


def _sq_err(y, p):
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def auc(y, score) -> float:
    """ROC AUC via the Mann-Whitney rank formula (ties count one half)."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    n1 = int(pos.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def f1(y, pred) -> float:
    """F1 score of the positive class (label 1)."""
    y = np.asarray(y)
    lab = _as_labels(pred)
    tp = float(np.sum((lab == 1) & (y == 1)))
    fp = float(np.sum((lab == 1) & (y != 1)))
    fn = float(np.sum((lab != 1) & (y == 1)))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean(_sq_err(y, p))))


def r2(y, p) -> float:
    y = np.asarray(y, dtype=float)
    ss_res = float(np.sum(_sq_err(y, p)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


@dataclass(frozen=True)
class Metric:
    name: str
    score: Callable[[np.ndarray, np.ndarray], float]
    greater_is_better: bool
    pointwise: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    needs_labels: bool = False  # True when McNemar's test makes sense

    @property
    def decomposable(self) -> bool:
        return self.pointwise is not None


def _mean_of(fn):
    return lambda y, p: float(np.mean(fn(y, p)))


METRICS: Dict[str, Metric] = {
    "accuracy": Metric("accuracy", _mean_of(_accuracy_pointwise), True, _accuracy_pointwise, needs_labels=True),
    "auc": Metric("auc", auc, True, None),
    "f1": Metric("f1", f1, True, None),
    "logloss": Metric("logloss", _mean_of(_logloss_pointwise), False, _logloss_pointwise),
    "brier": Metric("brier", _mean_of(_brier_pointwise), False, _brier_pointwise),
    "mae": Metric("mae", _mean_of(_abs_err), False, _abs_err),
    "mse": Metric("mse", _mean_of(_sq_err), False, _sq_err),
    "rmse": Metric("rmse", rmse, False, None),
    "r2": Metric("r2", r2, True, None),
}


def get_metric(metric) -> Metric:
    """Look up a metric by name, or pass a :class:`Metric` through."""
    if isinstance(metric, Metric):
        return metric
    try:
        return METRICS[str(metric).lower()]
    except KeyError as exc:
        raise ValueError(f"unknown metric {metric!r}; choose one of {sorted(METRICS)}") from exc
