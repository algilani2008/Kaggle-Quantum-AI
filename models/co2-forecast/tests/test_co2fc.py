import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from co2fc import HarmonicTrend, SeasonalNaive, SeasonalNaiveDrift, parse_csv  # noqa: E402
from co2fc.evaluate import (coverage, interval_offsets, mae, rolling_errors, split,  # noqa: E402
                            tune_rolling)


def synthetic(n_years=30, start="1990-01", slope=2.0, amp=3.0, noise=0.0, seed=0):
    """Clearly synthetic test series: linear trend + yearly sine (+ optional noise)."""
    idx = pd.period_range(start, periods=12 * n_years, freq="M")
    t = idx.year.to_numpy() + (idx.month.to_numpy() - 0.5) / 12
    rng = np.random.default_rng(seed)
    v = 300 + slope * (t - t[0]) + amp * np.sin(2 * np.pi * t) + noise * rng.standard_normal(len(t))
    return pd.Series(v, index=idx)


def make_csv(rows):
    head = "Date,Decimal Date,Average,Interpolated,Trend,Number of Days\n"
    return head + "\n".join(rows) + "\n"


def test_parse_reads_columns_by_position_and_flags_interpolated():
    text = make_csv(["1958-03,1958.2027,315.71,314.44,-01,-9.99,-0.99",
                     "1958-04,1958.2877,317.45,315.16,12,0.3,0.1"])
    df = parse_csv(text)
    assert list(df.index.astype(str)) == ["1958-03", "1958-04"]
    assert df["co2"].tolist() == [315.71, 317.45]
    assert df["co2_deseason"].tolist() == [314.44, 315.16]
    assert df["interpolated"].tolist() == [True, False]


def test_parse_rejects_gaps():
    text = make_csv(["2000-01,2000.04,370,369,20,0.3,0.1", "2000-03,2000.2,371,370,20,0.3,0.1"])
    with pytest.raises(ValueError, match="gaps"):
        parse_csv(text)


def test_seasonal_naive_repeats_last_year():
    y = synthetic(3)
    p = SeasonalNaive().fit(y).predict(30)
    assert np.allclose(p[:12], y.values[-12:])
    assert np.allclose(p[12:24], y.values[-12:])


def test_drift_recovers_exact_linear_trend():
    y = synthetic(12, slope=2.0)
    p = SeasonalNaiveDrift(drift_years=5).fit(y).predict(24)
    truth = synthetic(14, slope=2.0).values[-24:]
    assert np.allclose(p, truth, atol=1e-9)


def test_harmonic_trend_recovers_noise_free_signal():
    full = synthetic(25)
    y, future = full.iloc[:-36], full.iloc[-36:]
    m = HarmonicTrend(degree=1, harmonics=1, window_years=None).fit(y)
    assert mae(future.values, m.predict(36)) < 1e-6


def test_harmonic_trend_beats_naive_on_noisy_series():
    full = synthetic(25, noise=0.3, seed=1)
    y, future = full.iloc[:-36], full.iloc[-36:]
    learned = HarmonicTrend(degree=1, harmonics=2, window_years=15).fit(y).predict(36)
    naive = SeasonalNaive().fit(y).predict(36)
    assert mae(future.values, learned) < mae(future.values, naive)


def test_anchor_shifts_forecast_by_recent_residual():
    y = synthetic(20).copy()
    y.iloc[-6:] += 1.0  # recent level jump the smooth curve misses
    plain = HarmonicTrend(degree=1, harmonics=1, window_years=None).fit(y).predict(3)
    anch = HarmonicTrend(degree=1, harmonics=1, window_years=None, anchor_months=6).fit(y).predict(3)
    assert np.all(anch > plain)


def test_split_is_time_ordered_and_disjoint():
    y = synthetic(40)
    tr, va, te = split(y, "2019-12", "2022-12")
    assert tr.index.max() < va.index.min() and va.index.max() < te.index.min()
    assert len(tr) + len(va) + len(te) == len(y)


def test_rolling_errors_never_use_future():
    """Changing values after an origin's horizon must not change that origin's errors."""
    y = synthetic(30, noise=0.2, seed=2)
    e1 = rolling_errors(y, "harmonic_trend", {"degree": 1, "harmonics": 2, "window_years": 10},
                        h=12, first_origin="2000-12", step=12)
    y2 = y.copy()
    y2.iloc[-1] += 50  # only the very last month changes
    e2 = rolling_errors(y2, "harmonic_trend", {"degree": 1, "harmonics": 2, "window_years": 10},
                        h=12, first_origin="2000-12", step=12)
    assert np.allclose(e1[:-1], e2[:-1])


def test_tune_rolling_prefers_matching_structure():
    y = synthetic(30, amp=3.0, noise=0.05, seed=3) + 0.0
    space = {"degree": [1], "harmonics": [0, 1], "window_years": [10]}
    best, rows = tune_rolling(y, "harmonic_trend", space, h=12, first_origin="2005-12")
    assert best["harmonics"] == 1 and len(rows) == 2


def test_interval_offsets_and_coverage():
    errs = np.tile(np.linspace(-1, 1, 101)[:, None], (1, 3))
    lo, hi = interval_offsets(errs, 0.9)
    assert np.allclose(lo, -0.9) and np.allclose(hi, 0.9)
    assert coverage([0, 5], [-1, -1], [1, 1]) == 0.5


def test_train_end_to_end_on_synthetic_csv(tmp_path):
    y = synthetic(66, start="1958-03", slope=1.5, noise=0.2, seed=4)
    rows = [f"{p},{p.year + p.month / 12:.4f},{v:.2f},{v:.2f},20,0.3,0.1" for p, v in y.items()]
    csv = tmp_path / "syn.csv"
    csv.write_text(make_csv(rows))
    sys.path.insert(0, ROOT)
    import train
    out = tmp_path / "res"
    report = train.main(["--data", str(csv), "--out", str(out), "--no-plots"])
    assert set(report["test"]) == {"seasonal_naive", "seasonal_naive_drift", "harmonic_trend"}
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["split"]["test"][0] == "2023-01"
    fc = pd.read_csv(out / "forecast_next24.csv")
    assert len(fc) == 24 and (fc["lo90"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["hi90"]).all()
