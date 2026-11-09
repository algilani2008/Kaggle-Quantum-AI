import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from leakcheck import Thresholds, audit, infer_task  # noqa: E402
from leakcheck.cli import main as cli_main  # noqa: E402


def make_data(n=600, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.8 * X[:, 1] + rng.normal(0, 1.0, n) > 0).astype(int)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["color"] = rng.choice(["red", "green", "blue"], n)
    df["y"] = y
    return df


def split(df, seed=0):
    idx = np.random.RandomState(seed).permutation(len(df))
    cut = int(0.75 * len(df))
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]].drop(columns="y")


def checks(rep, sev="medium"):
    return {f.check for f in rep.at_least(sev)}


def test_clean_split_has_no_alarms():
    tr, te = split(make_data())
    rep = audit(tr, te, target="y")
    assert checks(rep) == set()
    assert rep.summary["adversarial_auc"] < 0.65


def test_target_copy_is_flagged_high():
    df = make_data()
    df["leak"] = df["y"]
    tr, te = split(df)
    rep = audit(tr, te, target="y")
    leak = [f for f in rep.findings if f.check == "target_leakage"]
    assert leak and leak[0].severity == "high" and leak[0].details["column"] == "leak"


def test_noisy_target_proxy_is_flagged():
    df = make_data()
    rng = np.random.RandomState(1)
    noisy = df["y"].to_numpy().copy()
    flip = rng.rand(len(df)) < 0.01  # 99% agreement with the label
    noisy[flip] = 1 - noisy[flip]
    df["proxy"] = noisy
    tr, te = split(df)
    assert "target_leakage" in checks(audit(tr, te, target="y"))


def test_regression_proxy_is_flagged():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"x": rng.normal(size=500)})
    df["y"] = 3 * df["x"] + rng.normal(size=500)
    df["y_plus_noise"] = df["y"] + rng.normal(0, 0.05, 500)
    tr, te = split(df)
    rep = audit(tr, te, target="y")
    assert rep.summary["task"] == "regression"
    assert any(f.details.get("column") == "y_plus_noise" for f in rep.at_least("medium"))


def test_sorted_row_id_is_high():
    df = make_data().sort_values("y").reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df)))
    tr, te = split(df)
    ids = [f for f in audit(tr, te, target="y").findings if f.check == "id_like_column"]
    assert ids and ids[0].severity == "high"


def test_random_id_is_not_high():
    df = make_data()
    df.insert(0, "uid", np.random.RandomState(3).permutation(len(df)))
    tr, te = split(df)
    ids = [f for f in audit(tr, te, target="y").findings if f.check == "id_like_column"]
    assert ids and ids[0].severity in ("low", "medium")


def test_cross_duplicates_detected():
    tr, te = split(make_data())
    te = pd.concat([te, tr.drop(columns="y").head(30)])
    rep = audit(tr, te, target="y")
    dup = [f for f in rep.findings if f.check == "cross_duplicates"]
    assert dup and dup[0].details["n"] == 30 and dup[0].severity == "high"


def test_shifted_test_set_detected():
    df = make_data(800)
    cut = df["c"] > df["c"].quantile(0.75)
    rep = audit(df[~cut], df[cut].drop(columns="y"), target="y")
    adv = [f for f in rep.findings if f.check == "adversarial_validation"][0]
    assert adv.severity == "high"
    assert next(iter(adv.details["top_features"])) == "c"


def test_tiny_sets_do_not_alarm_on_noise():
    df = make_data(60)
    tr, te = split(df)
    adv = [f for f in audit(tr, te, target="y").findings if f.check == "adversarial_validation"][0]
    assert adv.severity in ("info", "low")


def test_group_and_time_overlap():
    df = make_data(400)
    df["patient"] = np.arange(400) // 4
    df["date"] = pd.date_range("2024-01-01", periods=400, freq="D")
    tr, te = split(df)
    found = checks(audit(tr, te, target="y", group="patient", time="date"))
    assert {"group_overlap", "time_overlap"} <= found
    # a proper time split is clean on the time check
    tr2, te2 = df.iloc[:300], df.iloc[300:].drop(columns="y")
    assert "time_overlap" not in checks(audit(tr2, te2, target="y", time="date"))


def test_train_only_and_no_target():
    df = make_data()
    rep = audit(pd.concat([df, df.head(50)]))
    assert "train_duplicates" in checks(rep)
    assert rep.summary["task"] is None


def test_missing_values_and_categoricals_ok():
    df = make_data()
    df.loc[::7, "a"] = np.nan
    df.loc[::5, "color"] = None
    tr, te = split(df)
    audit(tr, te, target="y")  # must not raise


def test_bad_target_raises():
    with pytest.raises(ValueError):
        audit(make_data(), target="nope")


def test_infer_task():
    assert infer_task(pd.Series([0, 1, 1, 0])) == "classification"
    assert infer_task(pd.Series(["a", "b"])) == "classification"
    assert infer_task(pd.Series(np.linspace(0, 1, 100))) == "regression"


def test_thresholds_are_respected():
    df = make_data()
    df["leak"] = df["y"]
    tr, te = split(df)
    strict = Thresholds(leak_auc_high=1.01, leak_auc_medium=1.01, leak_bacc_high=1.01,
                        leak_bacc_medium=1.01)
    assert "target_leakage" not in checks(audit(tr, te, target="y", thresholds=strict))


def test_report_outputs():
    df = make_data()
    df["leak"] = df["y"]
    tr, te = split(df)
    rep = audit(tr, te, target="y")
    md = rep.to_markdown()
    assert "target_leakage" in md and "| high |" in md
    assert '"findings"' in rep.to_json()
    assert rep.worst == "high"


def test_cli(tmp_path):
    df = make_data()
    df["leak"] = df["y"]
    tr, te = split(df)
    tr.to_csv(tmp_path / "train.csv", index=False)
    te.to_csv(tmp_path / "test.csv", index=False)
    args = [str(tmp_path / "train.csv"), "--test", str(tmp_path / "test.csv"), "--target", "y",
            "--md", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json")]
    assert cli_main(args + ["--fail-on", "high"]) == 1
    assert cli_main(args) == 0
    assert (tmp_path / "r.md").read_text().startswith("# Leakage audit")
    # module entry point
    r = subprocess.run([sys.executable, "-m", "leakcheck", "--help"], cwd=os.path.dirname(HERE),
                       capture_output=True, text=True)
    assert r.returncode == 0 and "leakage" in r.stdout
