import numpy as np
import pytest
from sklearn.metrics import f1_score

from arsent.metrics import best_threshold, book_bootstrap, expected_calibration_error, summarize


def test_perfect_predictions():
    scores = summarize([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert scores["accuracy"] == 1 and scores["macro_f1"] == 1 and scores["roc_auc"] == 1


def test_majority_baseline_macro_f1_uses_both_classes():
    scores = summarize([0, 1, 1, 1], [1.0, 1.0, 1.0, 1.0])
    assert scores["macro_f1"] == pytest.approx(f1_score([0, 1, 1, 1], [1, 1, 1, 1], average="macro"))
    assert "roc_auc" not in scores


def test_ece_is_zero_when_probabilities_match_rates():
    y = np.array([0, 0, 0, 1] * 25)
    assert expected_calibration_error(y, np.full(100, 0.25)) == pytest.approx(0)


def test_ece_detects_overconfidence():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.95)) == pytest.approx(0.45)


def test_best_threshold_finds_the_gap():
    y = np.array([0] * 10 + [1] * 10)
    prob = np.r_[np.full(10, 0.6), np.full(10, 0.7)]
    assert 0.6 < best_threshold(y, prob) <= 0.7


def test_bootstrap_point_estimate_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    pred = np.where(rng.random(500) < 0.8, y, 1 - y)
    groups = rng.integers(0, 40, 500)
    out = book_bootstrap(y, pred, groups, reps=300)
    assert out["estimate"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert out["low"] < out["estimate"] < out["high"]


def test_bootstrap_difference_of_identical_models_is_zero():
    y = np.array([0, 1] * 30)
    pred = np.array([0, 1, 1, 1] * 15)
    groups = np.arange(60) % 6
    out = book_bootstrap(y, pred, groups, pred_b=pred, reps=100)
    assert out == {"estimate": 0, "low": 0, "high": 0}


def test_book_interval_is_wider_when_books_disagree():
    y = np.ones(400, dtype=int)
    y[::2] = 0
    pred = y.copy()
    bad = np.arange(200)  # half of the books are all wrong
    pred[bad] = 1 - pred[bad]
    books = np.arange(400) // 100
    clustered = book_bootstrap(y, pred, books, reps=500)
    single = book_bootstrap(y, pred, np.arange(400), reps=500)
    assert clustered["high"] - clustered["low"] > single["high"] - single["low"]
