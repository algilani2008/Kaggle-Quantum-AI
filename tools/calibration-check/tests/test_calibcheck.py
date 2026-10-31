import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibcheck import (  # noqa: E402
    IsotonicCalibrator, PlattScaling, TemperatureScaling, brier_decomposition,
    brier_score, calibration_report, expected_calibration_error, log_loss,
    maximum_calibration_error, reliability_bins,
)
from calibcheck.cli import main  # noqa: E402


def _perfect_binary(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return y, p


def _overconfident_multiclass(n=6000, k=4, t=3.0, seed=1):
    """Labels drawn from softmax(z); predictions are softmax(t*z) (over-confident)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.0, (n, k))
    true_p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in true_p])
    pred = np.exp(t * z) / np.exp(t * z).sum(1, keepdims=True)
    return y, pred


def test_ece_hand_example():
    # two bins (0,.5] and (.5,1]; binary top-label confidences
    y = np.array([1, 0, 1, 1])
    p = np.array([0.9, 0.9, 0.6, 0.2])  # p(class 1)
    # top-label: conf=[.9,.9,.6,.8], pred=[1,1,1,0], correct=[1,0,1,0]
    # all confidences in (0.5,1] -> one bin: mean conf=.8, acc=.5 -> ECE=.3
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.3)
    assert maximum_calibration_error(y, p, n_bins=2) == pytest.approx(0.3)


def test_perfectly_calibrated_has_small_ece():
    y, p = _perfect_binary()
    assert expected_calibration_error(y, p, 10) < 0.02


def test_constant_wrong_confidence():
    y = np.zeros(100, dtype=int)
    p = np.full(100, 0.8)  # always predicts class 1 with conf .8, always wrong
    assert expected_calibration_error(y, p, 10) == pytest.approx(0.8)


def test_brier_binary_and_multiclass():
    y = np.array([0, 1])
    assert brier_score(y, np.array([0.0, 1.0])) == 0.0
    assert brier_score(y, np.array([0.5, 0.5])) == pytest.approx(0.25)
    P = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
    assert brier_score(np.array([0, 1]), P) == pytest.approx(0.25)


def test_brier_decomposition_identity():
    y, p = _perfect_binary(5000)
    d = brier_decomposition(y, p, 10)
    assert d["reliability"] - d["resolution"] + d["uncertainty"] + d["within_bin"] == pytest.approx(d["brier"])
    assert d["reliability"] < 0.002


def test_bins_counts_sum_and_quantile():
    y, p = _perfect_binary(1000)
    b = reliability_bins(y, p, 10, "quantile")
    assert b.counts.sum() == 1000
    assert b.counts.max() - b.counts.min() <= 2


def test_input_validation():
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], np.array([[0.7, 0.7], [0.5, 0.5]]))
    with pytest.raises(ValueError):
        expected_calibration_error([0, 2], np.array([0.2, 0.3]))
    with pytest.raises(ValueError):
        brier_score([0, 1], np.array([1.2, 0.1]))


def test_temperature_scaling_fixes_overconfidence():
    y, p = _overconfident_multiclass()
    cal, ev = slice(0, 3000), slice(3000, None)
    ts = TemperatureScaling().fit(y[cal], p[cal])
    assert ts.temperature_ == pytest.approx(3.0, rel=0.2)
    q = ts.transform(p[ev])
    assert np.allclose(q.sum(1), 1)
    assert (q.argmax(1) == p[ev].argmax(1)).all()  # accuracy unchanged
    assert expected_calibration_error(y[ev], q) < 0.5 * expected_calibration_error(y[ev], p[ev])
    assert log_loss(y[ev], q) < log_loss(y[ev], p[ev])


def test_temperature_identity_on_calibrated():
    y, p = _perfect_binary(20000)
    ts = TemperatureScaling().fit(y, p)
    assert ts.temperature_ == pytest.approx(1.0, abs=0.08)
    assert ts.transform(p).shape == p.shape


def test_platt_and_isotonic_binary():
    rng = np.random.default_rng(3)
    true = rng.uniform(0.05, 0.95, 8000)
    y = (rng.uniform(size=8000) < true).astype(int)
    skewed = true ** 3  # badly calibrated but monotone
    for C in (PlattScaling, IsotonicCalibrator):
        m = C().fit(y[:4000], skewed[:4000])
        q = m.transform(skewed[4000:])
        assert q.ndim == 1
        assert expected_calibration_error(y[4000:], q) < 0.5 * expected_calibration_error(y[4000:], skewed[4000:])


def test_isotonic_multiclass_rows_sum_to_one():
    y, p = _overconfident_multiclass(3000)
    q = IsotonicCalibrator().fit(y, p).transform(p)
    assert np.allclose(q.sum(1), 1)


def test_platt_rejects_multiclass():
    y, p = _overconfident_multiclass(200)
    with pytest.raises(ValueError):
        PlattScaling().fit(y, p)


def test_report_fields():
    y, p = _overconfident_multiclass(500)
    r = calibration_report(y, p).to_dict()
    assert r["n_classes"] == 4 and r["n_samples"] == 500
    assert 0 <= r["ece"] <= 1 and r["mean_confidence"] > r["accuracy"]


def test_cli_end_to_end(tmp_path):
    y, p = _overconfident_multiclass(2000, k=3)
    df = pd.DataFrame(p, columns=["p0", "p1", "p2"])
    df["label"] = y
    csv = tmp_path / "preds.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "rep"
    assert main([str(csv), "--out-dir", str(out), "--calibrate", "temperature"]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["after"]["ece"] < rep["before"]["ece"]
    assert (out / "reliability.png").stat().st_size > 1000
    assert len(pd.read_csv(out / "bins.csv")) == 15


def test_cli_binary_single_column(tmp_path):
    y, p = _perfect_binary(1000)
    csv = tmp_path / "b.csv"
    pd.DataFrame({"y": y, "prob": p}).to_csv(csv, index=False)
    out = tmp_path / "r"
    assert main([str(csv), "--label-col", "y", "--prob-cols", "prob", "--out-dir", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert "brier_decomposition" in rep
