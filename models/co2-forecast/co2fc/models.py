"""Forecasting models for a monthly series with trend + yearly seasonality.

Every model follows the same tiny interface::

    m = Model(...).fit(y)      # y: pandas Series with a monthly PeriodIndex
    m.predict(h)               # numpy array with the next h monthly values

All models only look at data passed to ``fit``; nothing from the future leaks
in, which is what makes the time-ordered evaluation in ``train.py`` honest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PERIOD = 12


def _check(y: pd.Series, min_len: int) -> np.ndarray:
    v = np.asarray(y, dtype=float)
    if v.ndim != 1 or len(v) < min_len:
        raise ValueError(f"need a 1-D series with at least {min_len} values, got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("series contains NaN or inf")
    return v


@dataclass
class SeasonalNaive:
    """Baseline 1: next year's month = the same month last year."""

    name: str = "seasonal_naive"
    _last: np.ndarray | None = field(default=None, repr=False)

    def fit(self, y: pd.Series) -> "SeasonalNaive":
        self._last = _check(y, PERIOD)[-PERIOD:]
        return self

    def predict(self, h: int) -> np.ndarray:
        reps = int(np.ceil(h / PERIOD))
        return np.tile(self._last, reps)[:h]


@dataclass
class SeasonalNaiveDrift:
    """Baseline 2: same month last year + average yearly increase.

    The yearly increase is the mean 12-month change over the last
    ``drift_years`` years of the training data.
    """

    drift_years: int = 5
    name: str = "seasonal_naive_drift"
    _last: np.ndarray | None = field(default=None, repr=False)
    _drift: float = field(default=0.0, repr=False)

    def fit(self, y: pd.Series) -> "SeasonalNaiveDrift":
        n = PERIOD * (self.drift_years + 1)
        v = _check(y, n)
        diffs = v[PERIOD:] - v[:-PERIOD]
        self._drift = float(np.mean(diffs[-PERIOD * self.drift_years:]))
        self._last = v[-PERIOD:]
        return self

    def predict(self, h: int) -> np.ndarray:
        i = np.arange(h)
        years_ahead = i // PERIOD + 1
        return self._last[i % PERIOD] + years_ahead * self._drift


@dataclass
class HarmonicTrend:
    """Learned model: least-squares fit of a polynomial trend + Fourier seasonality.

    y(t) = sum_{p<=degree} a_p * s^p + sum_{k<=harmonics} [b_k sin(2 pi k t) + c_k cos(2 pi k t)]

    where t is time in years and s is t centred and scaled on the fit window.
    Only the last ``window_years`` years are used for fitting (``None`` = all),
    which lets the trend adapt to the recent growth rate.

    ``anchor_months > 0`` adds an intercept correction: forecasts are shifted by
    the mean fit residual over the last ``anchor_months`` months, so the forecast
    starts from where the series actually is rather than where the smooth
    curve says it should be.
    """

    degree: int = 2
    harmonics: int = 2
    window_years: int | None = 20
    anchor_months: int = 0
    ridge: float = 1e-8
    name: str = "harmonic_trend"
    _coef: np.ndarray | None = field(default=None, repr=False)
    _t_end: float = field(default=0.0, repr=False)
    _center: float = field(default=0.0, repr=False)
    _scale: float = field(default=1.0, repr=False)
    _shift: float = field(default=0.0, repr=False)

    def _features(self, t: np.ndarray) -> np.ndarray:
        s = (t - self._center) / self._scale
        cols = [s ** p for p in range(self.degree + 1)]
        for k in range(1, self.harmonics + 1):
            cols.append(np.sin(2 * np.pi * k * t))
            cols.append(np.cos(2 * np.pi * k * t))
        return np.column_stack(cols)

    @staticmethod
    def _time(index: pd.PeriodIndex) -> np.ndarray:
        # mid-month in decimal years: year + (month - 0.5) / 12
        return index.year.to_numpy() + (index.month.to_numpy() - 0.5) / PERIOD

    def fit(self, y: pd.Series) -> "HarmonicTrend":
        if not isinstance(y.index, pd.PeriodIndex):
            raise TypeError("HarmonicTrend needs a Series with a monthly PeriodIndex")
        if self.window_years is not None:
            y = y.iloc[-PERIOD * self.window_years:]
        v = _check(y, PERIOD * 2)
        t = self._time(y.index)
        self._center = float(t.mean())
        self._scale = float(max(t.max() - t.min(), 1e-9) / 2)
        self._t_end = float(t[-1])
        X = self._features(t)
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        self._coef = np.linalg.solve(A, X.T @ v)
        self._shift = 0.0
        if self.anchor_months > 0:
            resid = v - X @ self._coef
            self._shift = float(resid[-self.anchor_months:].mean())
        return self

    def predict(self, h: int) -> np.ndarray:
        t = self._t_end + np.arange(1, h + 1) / PERIOD
        return self._features(t) @ self._coef + self._shift


def make(name: str, **params):
    """Build a model by name (used by the CLI and the hyper-parameter search)."""
    table = {
        "seasonal_naive": SeasonalNaive,
        "seasonal_naive_drift": SeasonalNaiveDrift,
        "harmonic_trend": HarmonicTrend,
    }
    if name not in table:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(table)}")
    return table[name](**params)
