"""Post-hoc calibrators that work on predicted probabilities only.

They never need the original model: fit them on a held-out calibration
split (``fit(y_cal, probs_cal)``) and apply to new predictions
(``transform(probs_new)``). Output has the same shape as the input
(1-D for binary 1-D input, (n, K) otherwise).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from sklearn.isotonic import IsotonicRegression

from .metrics import _as_2d, _check_labels, EPS


def _restore_shape(p2d: np.ndarray, was_1d: bool) -> np.ndarray:
    return p2d[:, 1] if was_1d else p2d


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureScaling:
    """Divide log-probabilities by a single temperature T > 0 (Guo et al., 2017).

    T > 1 softens over-confident predictions, T < 1 sharpens them. Accuracy
    (argmax) is unchanged. Works for binary and multi-class input.
    """

    def __init__(self, bounds=(0.05, 20.0)):
        self.bounds = bounds
        self.temperature_ = None

    def fit(self, y_true, probs):
        p = _as_2d(probs)
        y = _check_labels(y_true, p.shape[1])
        logp = np.log(np.clip(p, EPS, 1.0))
        rows = np.arange(len(y))

        def nll(log_t):
            q = _softmax(logp / np.exp(log_t))
            return -np.mean(np.log(np.clip(q[rows, y], EPS, 1.0)))

        lo, hi = np.log(self.bounds[0]), np.log(self.bounds[1])
        res = minimize_scalar(nll, bounds=(lo, hi), method="bounded")
        self.temperature_ = float(np.exp(res.x))
        return self

    def transform(self, probs):
        if self.temperature_ is None:
            raise RuntimeError("call fit() first")
        was_1d = np.asarray(probs).ndim == 1
        p = _as_2d(probs)
        q = _softmax(np.log(np.clip(p, EPS, 1.0)) / self.temperature_)
        return _restore_shape(q, was_1d)


class PlattScaling:
    """Binary only: p' = sigmoid(a * logit(p) + b), fit by maximum likelihood."""

    def __init__(self):
        self.a_ = None
        self.b_ = None

    @staticmethod
    def _logit(f):
        f = np.clip(f, 1e-7, 1 - 1e-7)
        return np.log(f / (1 - f))

    def fit(self, y_true, probs):
        p = _as_2d(probs)
        if p.shape[1] != 2:
            raise ValueError("PlattScaling supports binary problems only")
        y = _check_labels(y_true, 2).astype(float)
        x = self._logit(p[:, 1])

        def nll(params):
            a, b = params
            z = a * x + b
            # numerically stable log(1 + exp(z)) - y*z
            return np.mean(np.logaddexp(0.0, z) - y * z)

        res = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
        self.a_, self.b_ = (float(v) for v in res.x)
        return self

    def transform(self, probs):
        if self.a_ is None:
            raise RuntimeError("call fit() first")
        was_1d = np.asarray(probs).ndim == 1
        p = _as_2d(probs)
        if p.shape[1] != 2:
            raise ValueError("PlattScaling supports binary problems only")
        f = 1.0 / (1.0 + np.exp(-(self.a_ * self._logit(p[:, 1]) + self.b_)))
        return _restore_shape(np.column_stack([1 - f, f]), was_1d)


class IsotonicCalibrator:
    """Monotone, non-parametric calibration (Zadrozny & Elkan, 2002).

    Binary: isotonic regression of y on p(class 1). Multi-class: one isotonic
    map per class (one-vs-rest), then rows are renormalised to sum to 1.
    Needs more calibration data than temperature scaling to avoid overfitting.
    """

    def __init__(self):
        self.models_ = None

    def fit(self, y_true, probs):
        p = _as_2d(probs)
        y = _check_labels(y_true, p.shape[1])
        k = p.shape[1]
        cols = [1] if k == 2 else range(k)
        self.models_ = {}
        for c in cols:
            m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            m.fit(p[:, c], (y == c).astype(float))
            self.models_[c] = m
        self.n_classes_ = k
        return self

    def transform(self, probs):
        if self.models_ is None:
            raise RuntimeError("call fit() first")
        was_1d = np.asarray(probs).ndim == 1
        p = _as_2d(probs)
        if p.shape[1] != self.n_classes_:
            raise ValueError("number of classes differs from fit()")
        if self.n_classes_ == 2:
            f = self.models_[1].predict(p[:, 1])
            q = np.column_stack([1 - f, f])
        else:
            q = np.column_stack([self.models_[c].predict(p[:, c]) for c in range(self.n_classes_)])
            s = q.sum(axis=1, keepdims=True)
            # if every per-class map returns 0 for a row, fall back to the input
            q = np.where(s > EPS, q / np.maximum(s, EPS), p)
        return _restore_shape(q, was_1d)


CALIBRATORS = {
    "temperature": TemperatureScaling,
    "platt": PlattScaling,
    "isotonic": IsotonicCalibrator,
}
