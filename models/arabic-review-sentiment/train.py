#!/usr/bin/env python3
"""Train and evaluate Arabic sentiment models on LABR.

Run from this folder:  python train.py
Quick smoke run:       python train.py --sample 6000 --audit-seeds 1

Writes metrics, tables and charts to results/ and the final model to
artifacts/model.joblib. The test set is only scored after the model, its
regularization and its decision threshold are fixed on validation data.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB

from arsent.dataset import deduplicate, load_reviews, official_split, to_binary
from arsent.metrics import best_threshold, book_bootstrap, summarize
from arsent.models import (
    FEATURE_SPECS, FeatureSpec, PlattCalibrator, SentimentModel, naive_bayes_features,
)
from arsent.splits import book_split, book_three_way, random_split
from arsent.textnorm import normalize_arabic
from plots import calibration_chart, comparison_chart, leakage_chart, length_chart

ROOT = Path(__file__).resolve().parent
C_GRID = (1.0, 3.0, 10.0, 30.0)
SEED = 42
LENGTH_BINS = [0, 50, 150, 400, 10**7]
LENGTH_NAMES = ["under 50", "50-149", "150-399", "400+"]


def logistic(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=300)


def select_model(train: pd.DataFrame, val: pd.DataFrame, specs=FEATURE_SPECS, grid=C_GRID):
    """Score every feature set and C on validation data. Test data stays out."""
    rows = []
    for spec in specs:
        started = time.time()
        features = spec.build()
        x_train, x_val = features.fit_transform(train["text"]), features.transform(val["text"])
        for c in grid:
            prob = logistic(c).fit(x_train, train["label"]).predict_proba(x_val)[:, 1]
            scores = summarize(val["label"], prob)
            rows.append({"features": spec.name, "C": c, "val_macro_f1": scores["macro_f1"],
                         "val_accuracy": scores["accuracy"], "val_roc_auc": scores["roc_auc"]})
        print(f"  {spec.name}: {time.time() - started:.0f}s", flush=True)
    return pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False, ignore_index=True)


def fit_model(spec: FeatureSpec, c: float, frame: pd.DataFrame) -> SentimentModel:
    features = spec.build()
    matrix = features.fit_transform(frame["text"])
    return SentimentModel(features, logistic(c).fit(matrix, frame["label"]))


def fit_naive_bayes(frame: pd.DataFrame) -> SentimentModel:
    features = naive_bayes_features()
    matrix = features.fit_transform(frame["text"])
    return SentimentModel(features, MultinomialNB().fit(matrix, frame["label"]))


def audit_one(kind: str, seed: int, data: pd.DataFrame, spec: FeatureSpec, c: float) -> dict:
    """Same model and sizes; only the way of splitting changes."""
    if kind == "book":
        train, test = book_split(data["book_id"], 0.2, seed)
    else:
        train, test = random_split(data["label"], 0.2, seed)
    model = fit_model(spec, c, data.iloc[train])
    scores = summarize(data["label"].iloc[test], model.predict_proba(data["text"].iloc[test]))
    return {"split": kind, "seed": seed, "macro_f1": scores["macro_f1"],
            "accuracy": scores["accuracy"], "test_rows": len(test)}


def audit_official(frame: pd.DataFrame, data_dir: Path, spec: FeatureSpec, c: float) -> dict:
    """Score the split published with LABR and count test reviews seen in training."""
    train_idx, test_idx = official_split(data_dir)
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    if (train["rating"] == 3).any() or (test["rating"] == 3).any():
        raise ValueError("official binary split should not contain 3-star reviews")
    train_keys = set(train["text"].map(normalize_arabic))
    repeated = test["text"].map(normalize_arabic).isin(train_keys)
    model = fit_model(spec, c, train)
    scores = summarize(test["label"], model.predict_proba(test["text"]))
    return {
        "split": "official", "macro_f1": scores["macro_f1"], "accuracy": scores["accuracy"],
        "train_rows": len(train), "test_rows": len(test),
        "test_rows_with_text_in_train": int(repeated.sum()),
        "test_share_with_text_in_train": float(repeated.mean()),
        "test_books_also_in_train_share": float(test["book_id"].isin(train["book_id"]).mean()),
    }


def by_length(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    length = test["text"].str.len()
    bucket = pd.cut(length, LENGTH_BINS, right=False, labels=LENGTH_NAMES)
    rows = []
    for name in LENGTH_NAMES:
        mask = (bucket == name).to_numpy()
        rows.append({
            "characters": name, "reviews": int(mask.sum()),
            "macro_f1": f1_score(test["label"][mask], pred[mask], average="macro"),
            "accuracy": float((test["label"][mask].to_numpy() == pred[mask]).mean()),
        })
    return pd.DataFrame(rows)


def top_features(model: SentimentModel, count: int = 25) -> pd.DataFrame:
    names = np.array(model.features.feature_names())
    weights = model.classifier.coef_[0]
    order = np.argsort(weights)
    rows = [{"direction": "negative", "feature": names[i], "weight": weights[i]} for i in order[:count]]
    rows += [{"direction": "positive", "feature": names[i], "weight": weights[i]} for i in order[::-1][:count]]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--sample", type=int, help="use only this many reviews (smoke runs)")
    parser.add_argument("--audit-seeds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = load_reviews(args.data_dir)
    labelled = to_binary(raw)
    data, dedupe_report = deduplicate(labelled)
    if args.sample:
        data = data.sample(args.sample, random_state=SEED).reset_index(drop=True)
    train_pos, val_pos, test_pos = book_three_way(data["book_id"], SEED)
    train, val, test = (data.iloc[p].reset_index(drop=True) for p in (train_pos, val_pos, test_pos))
    assert not (set(train["book_id"]) | set(val["book_id"])) & set(test["book_id"])
    print(f"train {len(train)}  val {len(val)}  test {len(test)}", flush=True)

    # Model selection and threshold: validation data only.
    grid = select_model(train, val)
    grid.to_csv(args.out / "validation_grid.csv", index=False, float_format="%.4f")
    best = grid.iloc[0]
    spec = next(s for s in FEATURE_SPECS if s.name == best["features"])
    c = float(best["C"])
    print(f"selected {spec.name} C={c}", flush=True)
    selection_model = fit_model(spec, c, train)
    val_raw = selection_model.predict_proba(val["text"])
    calibrator = PlattCalibrator().fit(val_raw, val["label"])
    threshold = best_threshold(val["label"], calibrator.transform(val_raw))

    # Refit on train + validation, then score the test set once.
    development = pd.concat([train, val], ignore_index=True)
    final = fit_model(spec, c, development)
    final.threshold = threshold
    final.calibrator = calibrator
    naive_bayes = fit_naive_bayes(development)
    y_test = test["label"].to_numpy()
    groups = test["book_id"].to_numpy()
    raw_scores = final.raw_scores(test["text"])
    prob = calibrator.transform(raw_scores)
    pred = (prob >= threshold).astype(int)
    prob_nb = naive_bayes.predict_proba(test["text"])
    pred_nb = (prob_nb >= 0.5).astype(int)
    majority_class = int(development["label"].mode()[0])
    pred_majority = np.full(len(test), majority_class)

    test_scores = {
        "majority_class": summarize(y_test, np.full(len(test), float(majority_class))),
        "naive_bayes_words": summarize(y_test, prob_nb),
        "final_model": summarize(y_test, prob, threshold),
        "final_model_threshold_0.5": summarize(y_test, prob),
        "final_model_uncalibrated": summarize(y_test, raw_scores),
    }
    intervals = {
        "final_macro_f1": book_bootstrap(y_test, pred, groups),
        "naive_bayes_macro_f1": book_bootstrap(y_test, pred_nb, groups),
        "majority_macro_f1": book_bootstrap(y_test, pred_majority, groups),
        "final_minus_naive_bayes": book_bootstrap(y_test, pred, groups, pred_nb),
    }

    # Three-star reviews from test books: no true label, shown for transparency.
    neutral = raw[(raw["rating"] == 3) & raw["book_id"].isin(set(test["book_id"]))]
    if len(neutral):
        neutral = neutral.drop_duplicates("text")
        neutral_prob = final.predict_proba(neutral["text"])
    neutral_report = {
        "reviews": len(neutral),
        "share_called_positive": float((neutral_prob >= threshold).mean()) if len(neutral) else None,
    }

    lengths = by_length(test, pred)
    lengths.to_csv(args.out / "by_review_length.csv", index=False, float_format="%.4f")
    top_features(final).to_csv(args.out / "top_features.csv", index=False, float_format="%.3f")

    # Leakage audit: the same chosen model under three ways of splitting.
    # Published indices count rows of reviews.tsv, so they need the unfiltered frame.
    official = None
    if not args.sample:
        official = audit_official(raw.assign(label=(raw["rating"] >= 4).astype(int)), args.data_dir, spec, c)
    tasks = [(k, s) for k in ("random", "book") for s in range(args.audit_seeds)]
    audit_rows = joblib.Parallel(n_jobs=args.jobs)(
        joblib.delayed(audit_one)(k, s, data, spec, c) for k, s in tasks
    )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.out / "leakage_audit.csv", index=False, float_format="%.4f")

    joblib.dump(final, _artifact_path())
    summary = {
        "dataset": {
            "rows_raw": len(raw), "rows_binary": len(labelled), **dedupe_report,
            "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
            "train_books": int(train["book_id"].nunique()),
            "val_books": int(val["book_id"].nunique()),
            "test_books": int(test["book_id"].nunique()),
            "positive_share_test": float(y_test.mean()),
            "positive_share_dev": float(development["label"].mean()),
        },
        "selected": {"features": spec.name, "C": c, "threshold": threshold,
                     "val_macro_f1": float(best["val_macro_f1"])},
        "test": test_scores,
        "intervals_95_book_bootstrap": intervals,
        "three_star_reviews": neutral_report,
        "leakage_audit": {
            "official_split": official,
            "random_mean": audit[audit.split == "random"]["macro_f1"].mean(),
            "random_std": audit[audit.split == "random"]["macro_f1"].std(),
            "book_mean": audit[audit.split == "book"]["macro_f1"].mean(),
            "book_std": audit[audit.split == "book"]["macro_f1"].std(),
        },
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "seed": SEED, "seconds": round(time.time() - started)},
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    comparison_chart(summary, args.out / "model_comparison.png")
    leakage_chart(audit, official, args.out / "leakage_audit.png")
    calibration_chart(y_test, prob, args.out / "calibration.png")
    length_chart(lengths, args.out / "by_review_length.png")
    print(json.dumps({k: summary[k] for k in ("selected", "test", "leakage_audit")}, indent=2, default=float))


def _artifact_path() -> Path:
    path = ROOT / "artifacts" / "model.joblib"
    path.parent.mkdir(exist_ok=True)
    return path


if __name__ == "__main__":
    main()
