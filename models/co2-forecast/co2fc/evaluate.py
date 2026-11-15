"""Metrics, time-ordered splits, rolling-origin backtests and prediction intervals."""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd

from .models import PERIOD, make


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mape(y, p) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean(np.abs((y - np.asarray(p)) / y)) * 100)


def coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def scores(y, p) -> dict:
    return {"mae": mae(y, p), "rmse": rmse(y, p), "mape_pct": mape(y, p)}


def split(y: pd.Series, train_end: str, val_end: str):
    """Time-ordered split: train <= train_end < val <= val_end < test."""
    te, ve = pd.Period(train_end, "M"), pd.Period(val_end, "M")
    if not te < ve < y.index[-1]:
        raise ValueError("need train_end < val_end < last month of the series")
    return y[y.index <= te], y[(y.index > te) & (y.index <= ve)], y[y.index > ve]


def forecast_from(y_hist: pd.Series, name: str, params: dict, h: int) -> np.ndarray:
    return make(name, **params).fit(y_hist).predict(h)


def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, vals)) for vals in itertools.product(*space.values())]


def tune(train: pd.Series, val: pd.Series, name: str, space: dict,
         metric: Callable = mae) -> tuple[dict, list[dict]]:
    """Pick the params that minimise ``metric`` on the validation window.

    The model is fitted on ``train`` only and forecasts the whole validation
    window from its end (one forecast origin, horizon = len(val)).
    """
    rows = []
    for params in grid(space):
        try:
            pred = forecast_from(train, name, params, len(val))
        except ValueError:
            continue  # e.g. window longer than available history
        rows.append({**params, "val_mae": metric(val.values, pred)})
    if not rows:
        raise ValueError("no parameter combination could be fitted")
    best = min(rows, key=lambda r: r["val_mae"])
    best_params = {k: best[k] for k in space}
    return best_params, rows


def tune_rolling(y: pd.Series, name: str, space: dict, h: int, first_origin: str,
                 step: int = 6) -> tuple[dict, list[dict]]:
    """Time-series cross-validation: pick params with the lowest mean absolute
    error over many forecast origins (see ``rolling_errors``). Every target month
    used here lies inside ``y``, so pass train+validation only, never the test.
    """
    rows = []
    for params in grid(space):
        try:
            err = rolling_errors(y, name, params, h, first_origin, step)
        except ValueError:
            continue
        rows.append({**params, "cv_mae": float(np.mean(np.abs(err))), "origins": int(len(err))})
    if not rows:
        raise ValueError("no parameter combination could be fitted")
    best = min(rows, key=lambda r: r["cv_mae"])
    return {k: best[k] for k in space}, rows


def rolling_errors(y: pd.Series, name: str, params: dict, h: int,
                   first_origin: str, step: int = 6) -> np.ndarray:
    """Forecast errors (actual - predicted) from many past origins.

    Returns an array of shape (n_origins, h). Origins run every ``step`` months
    from ``first_origin`` up to the last one whose full horizon is observed.
    """
    start = y.index.get_loc(pd.Period(first_origin, "M"))
    errs = []
    for o in range(start, len(y) - h, step):
        hist, fut = y.iloc[: o + 1], y.iloc[o + 1: o + 1 + h]
        errs.append(fut.values - forecast_from(hist, name, params, h))
    if not errs:
        raise ValueError("no complete backtest origin; move first_origin earlier")
    return np.vstack(errs)


def interval_offsets(errors: np.ndarray, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Empirical per-horizon lower/upper error quantiles for a central interval."""
    a = (1 - level) / 2
    return np.quantile(errors, a, axis=0), np.quantile(errors, 1 - a, axis=0)


def by_horizon_year(y, p) -> list[dict]:
    """MAE per forecast year (months 1-12, 13-24, ...)."""
    y, p = np.asarray(y), np.asarray(p)
    out = []
    for k in range(int(np.ceil(len(y) / PERIOD))):
        sl = slice(k * PERIOD, (k + 1) * PERIOD)
        out.append({"horizon_year": k + 1, "months": int(len(y[sl])), "mae": mae(y[sl], p[sl])})
    return out
