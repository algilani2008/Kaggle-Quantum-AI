import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import sigdiff as sd  # noqa: E402
from sigdiff.metrics import auc, f1  # noqa: E402


def _toy(n=600, gap=0.0, seed=1):
    """Binary labels and two noisy probability predictions; B gets `gap` extra signal."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    z = rng.normal(size=n)
    pa = 1 / (1 + np.exp(-(1.0 * (2 * y - 1) + z + rng.normal(size=n))))
    pb = 1 / (1 + np.exp(-((1.0 + gap) * (2 * y - 1) + z + rng.normal(size=n))))
    return y, pa, pb


# ---------------------------------------------------------------- metrics
def test_auc_matches_sklearn_formula_with_ties():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.4, 0.8, 0.3, 0.9])
    # 9 pos/neg pairs: (0.4 vs 0.4) is a tie -> 8.5 / 9
    assert auc(y, s) == pytest.approx(8.5 / 9)


def test_f1_and_accuracy_from_probabilities():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.7, 0.1])
    assert f1(y, p) == pytest.approx(0.5)
    assert sd.get_metric("accuracy").score(y, p) == pytest.approx(0.5)


def test_pointwise_means_equal_scores():
    y, pa, _ = _toy()
    for name, m in sd.METRICS.items():
        if m.decomposable:
            assert np.mean(m.pointwise(y, pa)) == pytest.approx(m.score(y, pa)), name


def test_unknown_metric():
    with pytest.raises(ValueError):
        sd.get_metric("nope")


# ---------------------------------------------------------------- paired tests
def test_identical_predictions_give_zero_and_p_one():
    y, pa, _ = _toy()
    r = sd.paired_bootstrap(y, pa, pa, "logloss", n_resamples=200)
    assert r.diff == 0 and r.ci_low == 0 and r.ci_high == 0
    assert sd.permutation_test(y, pa, pa, "auc", n_resamples=200).p_value == 1.0
    assert sd.mcnemar(y, pa, pa).p_value == 1.0


def test_clear_improvement_is_detected_by_all_methods():
    y, pa, pb = _toy(n=2000, gap=1.5)
    for metric in ["accuracy", "auc", "logloss", "brier", "f1"]:
        c = sd.compare(y, pa, pb, metric, n_resamples=500)
        assert c.p_value < 0.01, metric
        assert c.verdict.startswith("B is better"), metric
        assert c.ci_low <= c.diff <= c.ci_high
    assert sd.permutation_test(y, pa, pb, "auc", n_resamples=300).p_value < 0.01
    assert sd.mcnemar(y, pa, pb).p_value < 0.01


def test_direction_for_lower_is_better_metric():
    y, pa, pb = _toy(n=2000, gap=1.5)
    c = sd.compare(y, pb, pa, "logloss", n_resamples=300, name_a="good", name_b="bad")
    assert c.diff > 0  # logloss of the worse model is higher
    assert c.verdict.startswith("good is better")


def test_vectorised_and_loop_bootstrap_agree():
    """Decomposable metrics use a fast path; check it against the generic loop."""
    y, pa, pb = _toy(n=400, gap=0.3)
    fast = sd.paired_bootstrap(y, pa, pb, "brier", n_resamples=3000, seed=5)
    m = sd.get_metric("brier")
    slow_metric = sd.Metric("brier_slow", m.score, False, None)
    slow = sd.paired_bootstrap(y, pa, pb, slow_metric, n_resamples=3000, seed=6)
    assert fast.diff == pytest.approx(slow.diff)
    assert fast.ci_low == pytest.approx(slow.ci_low, abs=0.004)
    assert fast.ci_high == pytest.approx(slow.ci_high, abs=0.004)


def test_mcnemar_exact_value():
    y = np.ones(20, dtype=int)
    a = np.array([1] * 10 + [0] * 10)
    b = np.array([1] * 10 + [1] * 8 + [0] * 2)  # B right on 8 rows where A is wrong
    r = sd.mcnemar(y, a, b)
    assert r.p_value == pytest.approx(stats.binomtest(8, 8, 0.5).pvalue)
    assert r.diff == pytest.approx(8 / 20)


def test_permutation_pvalue_is_roughly_uniform_under_null():
    """Exchangeable predictions: rejection rate at 5% should be near 5%."""
    rejections = 0
    trials = 150
    for s in range(trials):
        y, pa, pb = _toy(n=200, gap=0.0, seed=100 + s)
        rejections += sd.permutation_test(y, pa, pb, "logloss", n_resamples=200, seed=s).p_value < 0.05
    assert rejections / trials < 0.12


def test_groups_widen_interval_when_rows_are_clustered():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(40), 25)                    # 40 clusters of 25 rows
    shift = rng.normal(scale=0.3, size=40)[g]           # cluster-level difference in error
    y = np.zeros(g.size)
    pa = np.abs(rng.normal(size=g.size))
    pb = np.abs(pa + shift + 0.02 * rng.normal(size=g.size))
    flat = sd.paired_bootstrap(y, pa, pb, "mae", n_resamples=800)
    clus = sd.paired_bootstrap(y, pa, pb, "mae", n_resamples=800, groups=g)
    assert (clus.ci_high - clus.ci_low) > 2 * (flat.ci_high - flat.ci_low)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        sd.compare([0, 1, 1], [0.2, 0.8], [0.1, 0.9, 0.3])


def test_mcnemar_rejects_non_accuracy():
    y, pa, pb = _toy()
    with pytest.raises(ValueError):
        sd.compare(y, pa, pb, "auc", method="mcnemar")


# ---------------------------------------------------------------- multiple comparisons
def test_holm_known_values():
    adj = sd.holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_compare_many_uses_holm():
    y, pa, pb = _toy(n=1500, gap=1.5)
    res = sd.compare_many(y, {"base": pa, "strong": pb, "same": pa.copy()}, "base", "auc", n_resamples=300)
    names = {r.name_b: r for r in res}
    assert names["strong"].verdict.startswith("strong is better")
    assert names["same"].p_value == 1.0
    assert all(r.method.endswith("+holm") for r in res)


# ---------------------------------------------------------------- CV tests
def test_corrected_ttest_formula():
    a = np.array([0.80, 0.82, 0.79, 0.81, 0.83, 0.80, 0.78, 0.82, 0.81, 0.80])
    b = a + np.array([0.02, 0.01, 0.03, 0.00, 0.02, 0.01, 0.02, 0.01, 0.03, 0.02])
    d = b - a
    r = sd.corrected_resampled_ttest(a, b, n_train=900, n_test=100)
    se = np.sqrt(d.var(ddof=1) * (1 / 10 + 100 / 900))
    t = d.mean() / se
    assert r.p_value == pytest.approx(2 * stats.t.sf(abs(t), 9))
    naive = sd.naive_ttest(a, b)
    assert naive.p_value == pytest.approx(stats.ttest_rel(b, a).pvalue)
    assert r.p_value > naive.p_value  # the correction is more cautious


def test_five_by_two_formula():
    rng = np.random.default_rng(3)
    a = rng.uniform(0.7, 0.8, size=(5, 2))
    b = a + rng.normal(0.01, 0.01, size=(5, 2))
    d = b - a
    s2 = ((d - d.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    t = d[0, 0] / np.sqrt(s2.mean())
    r = sd.five_by_two_cv_ttest(a, b)
    assert r.p_value == pytest.approx(2 * stats.t.sf(abs(t), 5))
    with pytest.raises(ValueError):
        sd.five_by_two_cv_ttest(a.ravel(), b.ravel())


# ---------------------------------------------------------------- CLI
def _run(args, cwd):
    return subprocess.run([sys.executable, "-m", "sigdiff", *args], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": ROOT})


def test_cli_compare_and_cv(tmp_path):
    y, pa, pb = _toy(n=800, gap=1.5)
    pd.DataFrame({"y": y, "a": pa, "b": pb, "c": pa}).to_csv(tmp_path / "p.csv", index=False)
    out = _run(["compare", str(tmp_path / "p.csv"), "--target", "y", "--a", "a", "--b", "b",
                "--metric", "auc", "--n-resamples", "300", "--json"], tmp_path)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout)
    assert res[0]["verdict"].startswith("b is better")

    out = _run(["compare", str(tmp_path / "p.csv"), "--target", "y", "--a", "a", "--b", "b,c",
                "--n-resamples", "200"], tmp_path)
    assert out.returncode == 0 and "holm" in out.stdout

    out = _run(["compare", str(tmp_path / "p.csv"), "--target", "y", "--a", "a", "--b", "zzz"], tmp_path)
    assert out.returncode == 2

    pd.DataFrame({"a": np.linspace(0.7, 0.8, 10), "b": np.linspace(0.71, 0.82, 10)}).to_csv(tmp_path / "f.csv", index=False)
    out = _run(["cv", str(tmp_path / "f.csv"), "--a", "a", "--b", "b", "--n-train", "900", "--n-test", "100",
                "--five-by-two"], tmp_path)
    assert out.returncode == 0, out.stderr
    assert "corrected_resampled_ttest" in out.stdout and "five_by_two" in out.stdout
