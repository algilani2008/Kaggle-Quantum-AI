#!/usr/bin/env python3
"""
Kaggle Auto-ML Competition Engine — Advanced Stacking Pipeline
================================================================

Steps (each run):
  1. Load train / test / sample_submission CSVs from --data-dir and infer the
     target column, ID column and task type (binary, multiclass, regression).
  2. Feature engineering driven by the data's dimensions: encode categoricals,
     add row statistics, then generate cross features (product, ratio,
     difference) among the most informative numeric columns.
  3. Train XGBoost, LightGBM and CatBoost with K-fold out-of-fold predictions
     and stack them with a linear meta-model.
  4. Error analysis on the worst 10% out-of-fold predictions; use the result to
     choose a second round of hyperparameters (and hard-example weighting),
     keeping whichever round has the better CV score.
  5. Compare the final CV score with the baseline (best previous run stored in
     the state file, or a default single-LightGBM score on the first run) and
     write submission.csv only if it beats the baseline.
  6. Write report.json and report.md summarising features, steps and scores.

Usage:
  python pipeline.py --data-dir data --out-dir output --state-file state.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 42
VERSION = "1.0"


# --------------------------------------------------------------------------- #
# Data loading and inference
# --------------------------------------------------------------------------- #
def find_csv(data_dir: str, keywords: list[str]) -> str | None:
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".csv"))
    for kw in keywords:
        for f in files:
            if kw in f.lower():
                return os.path.join(data_dir, f)
    return None


def load_data(data_dir: str, target_arg: str | None, id_arg: str | None):
    train_p = find_csv(data_dir, ["train"])
    test_p = find_csv(data_dir, ["test"])
    sub_p = find_csv(data_dir, ["sample_submission", "submission", "sample"])
    if not train_p or not test_p:
        raise FileNotFoundError(
            f"Need train*.csv and test*.csv in {data_dir}; found "
            f"{os.listdir(data_dir)}"
        )
    train = pd.read_csv(train_p)
    test = pd.read_csv(test_p)
    sub = pd.read_csv(sub_p) if sub_p else None

    id_col = id_arg
    if id_col is None:
        if sub is not None and sub.columns[0] in test.columns:
            id_col = sub.columns[0]
        else:
            cands = [c for c in test.columns if c.lower() in ("id", "row_id", "index")]
            id_col = cands[0] if cands else None

    target = target_arg
    if target is None:
        if sub is not None and len(sub.columns) == 2 and sub.columns[1] in train.columns:
            target = sub.columns[1]
        else:
            extra = [c for c in train.columns if c not in test.columns]
            if len(extra) != 1:
                raise ValueError(
                    f"Cannot infer target column (candidates: {extra}). "
                    "Pass --target."
                )
            target = extra[0]
    return train, test, sub, id_col, target, (train_p, test_p, sub_p)


def infer_task(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "binary" if y.nunique() == 2 else "multiclass"
    n = y.nunique()
    if n == 2:
        return "binary"
    if n <= 20 and np.allclose(y.dropna(), y.dropna().round()):
        return "multiclass"
    return "regression"


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(train, test, target, id_col, task, y_enc, log):
    drop = [c for c in [target, id_col] if c]
    X = train.drop(columns=[c for c in drop if c in train.columns])
    Xt = test.drop(columns=[c for c in drop if c in test.columns])
    Xt = Xt[X.columns]
    n_rows, n_cols = X.shape
    feats_log = {"input_shape": [int(n_rows), int(n_cols)]}

    # Categorical encoding (frequency + ordinal codes)
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    for c in cat_cols:
        both = pd.concat([X[c], Xt[c]]).astype(str)
        freq = both.value_counts(normalize=True)
        codes = {v: i for i, v in enumerate(both.unique())}
        X[c + "__freq"] = X[c].astype(str).map(freq).astype(float)
        Xt[c + "__freq"] = Xt[c].astype(str).map(freq).astype(float)
        X[c] = X[c].astype(str).map(codes).astype(int)
        Xt[c] = Xt[c].astype(str).map(codes).astype(int)
    feats_log["categorical_encoded"] = cat_cols

    num_cols = [c for c in X.columns if c not in cat_cols and not c.endswith("__freq")]

    # Row statistics
    if len(num_cols) >= 3:
        for df in (X, Xt):
            v = df[num_cols].astype(float)
            df["row__mean"] = v.mean(axis=1)
            df["row__std"] = v.std(axis=1)
            df["row__nan"] = v.isna().sum(axis=1)
        feats_log["row_stats"] = ["row__mean", "row__std", "row__nan"]

    # Cross features: budget scales with data dimensions
    budget_k = int(np.clip(np.sqrt(max(n_cols, 1)) * 2, 3, 10))
    if n_rows * n_cols > 5_000_000:
        budget_k = min(budget_k, 5)
    created = []
    if len(num_cols) >= 2:
        sample = X[num_cols].fillna(X[num_cols].median())
        idx = np.random.RandomState(SEED).choice(len(sample), min(len(sample), 20000), replace=False)
        mi_fn = mutual_info_regression if task == "regression" else mutual_info_classif
        mi = mi_fn(sample.iloc[idx], y_enc[idx], random_state=SEED)
        top = [num_cols[i] for i in np.argsort(mi)[::-1][:budget_k]]
        feats_log["top_numeric_by_mutual_info"] = top
        for a, b in itertools.combinations(top, 2):
            for df in (X, Xt):
                fa, fb = df[a].astype(float), df[b].astype(float)
                df[f"{a}__x__{b}"] = fa * fb
                df[f"{a}__div__{b}"] = fa / (fb.replace(0, np.nan))
                df[f"{a}__minus__{b}"] = fa - fb
            created += [f"{a}__x__{b}", f"{a}__div__{b}", f"{a}__minus__{b}"]
    feats_log["cross_features"] = created
    feats_log["cross_feature_budget_k"] = budget_k

    for df in (X, Xt):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop constant columns
    const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X.drop(columns=const, inplace=True)
    Xt.drop(columns=const, inplace=True)
    feats_log["dropped_constant"] = const
    feats_log["final_shape"] = [int(X.shape[0]), int(X.shape[1])]
    log(f"Features: {n_cols} -> {X.shape[1]} ({len(created)} cross features)")
    return X, Xt, feats_log


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_models(task, n_classes, p, n_rows):
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    n_est = p["n_estimators"]
    common = dict(n_estimators=n_est, learning_rate=p["learning_rate"], random_state=SEED)
    if task == "regression":
        return {
            "xgboost": xgb.XGBRegressor(max_depth=p["max_depth"], subsample=p["subsample"],
                                        colsample_bytree=p["colsample"], min_child_weight=p["min_child_weight"],
                                        reg_lambda=p["reg_lambda"], tree_method="hist", n_jobs=-1, **common),
            "lightgbm": lgb.LGBMRegressor(num_leaves=p["num_leaves"], subsample=p["subsample"], subsample_freq=1,
                                          colsample_bytree=p["colsample"], min_child_samples=p["min_child_samples"],
                                          reg_lambda=p["reg_lambda"], verbose=-1, n_jobs=-1, **common),
            "catboost": cb.CatBoostRegressor(iterations=n_est, learning_rate=p["learning_rate"],
                                             depth=min(p["max_depth"], 10), l2_leaf_reg=p["reg_lambda"],
                                             random_seed=SEED, verbose=0, thread_count=-1),
        }
    obj_x = "binary:logistic" if task == "binary" else "multi:softprob"
    return {
        "xgboost": xgb.XGBClassifier(max_depth=p["max_depth"], subsample=p["subsample"],
                                     colsample_bytree=p["colsample"], min_child_weight=p["min_child_weight"],
                                     reg_lambda=p["reg_lambda"], objective=obj_x, tree_method="hist",
                                     n_jobs=-1, eval_metric="logloss", **common),
        "lightgbm": lgb.LGBMClassifier(num_leaves=p["num_leaves"], subsample=p["subsample"], subsample_freq=1,
                                       colsample_bytree=p["colsample"], min_child_samples=p["min_child_samples"],
                                       reg_lambda=p["reg_lambda"], verbose=-1, n_jobs=-1, **common),
        "catboost": cb.CatBoostClassifier(iterations=n_est, learning_rate=p["learning_rate"],
                                          depth=min(p["max_depth"], 10), l2_leaf_reg=p["reg_lambda"],
                                          random_seed=SEED, verbose=0, thread_count=-1),
    }


def predict(model, X, task):
    if task == "regression":
        return model.predict(X)
    proba = model.predict_proba(X)
    return proba[:, 1] if task == "binary" else proba


def score(task, y, pred):
    """Returns (score, higher_is_better, metric_name)."""
    if task == "binary":
        return roc_auc_score(y, pred), True, "roc_auc"
    if task == "multiclass":
        return log_loss(y, np.clip(pred, 1e-15, 1)), False, "log_loss"
    return float(np.sqrt(mean_squared_error(y, pred))), False, "rmse"


def per_row_error(task, y, pred):
    if task == "regression":
        return np.abs(y - pred)
    if task == "binary":
        p = np.clip(pred, 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return -np.log(np.clip(pred[np.arange(len(y)), y], 1e-15, 1))


def run_stack(X, y, Xt, task, n_classes, params, n_folds, weights, log):
    folds = (KFold(n_folds, shuffle=True, random_state=SEED) if task == "regression"
             else StratifiedKFold(n_folds, shuffle=True, random_state=SEED))
    names = ["xgboost", "lightgbm", "catboost"]
    shape_o = (len(X),) if task != "multiclass" else (len(X), n_classes)
    shape_t = (len(Xt),) if task != "multiclass" else (len(Xt), n_classes)
    oof = {n: np.zeros(shape_o) for n in names}
    tst = {n: np.zeros(shape_t) for n in names}
    for k, (tr, va) in enumerate(folds.split(X, y)):
        models = make_models(task, n_classes, params, len(X))
        for n, m in models.items():
            w = weights[tr] if weights is not None else None
            m.fit(X.iloc[tr], y[tr], sample_weight=w)
            oof[n][va] = predict(m, X.iloc[va], task)
            tst[n] += predict(m, Xt, task) / n_folds
        log(f"  fold {k + 1}/{n_folds} done")

    base_scores = {n: score(task, y, oof[n])[0] for n in names}

    # Meta-model on out-of-fold predictions
    def stack_mat(d):
        return np.column_stack([d[n] if d[n].ndim == 1 else d[n] for n in names])

    M, Mt = stack_mat(oof), stack_mat(tst)
    meta_oof = np.zeros(shape_o)
    for tr, va in folds.split(M, y):
        meta = Ridge(alpha=1.0) if task == "regression" else LogisticRegression(C=1.0, max_iter=2000)
        meta.fit(M[tr], y[tr])
        meta_oof[va] = meta.predict(M[va]) if task == "regression" else (
            meta.predict_proba(M[va])[:, 1] if task == "binary" else meta.predict_proba(M[va]))
    meta = Ridge(alpha=1.0) if task == "regression" else LogisticRegression(C=1.0, max_iter=2000)
    meta.fit(M, y)
    meta_test = meta.predict(Mt) if task == "regression" else (
        meta.predict_proba(Mt)[:, 1] if task == "binary" else meta.predict_proba(Mt))
    stack_score = score(task, y, meta_oof)[0]
    return dict(oof=meta_oof, test=meta_test, base_scores=base_scores, stack_score=stack_score)


def error_analysis(X, y, oof, task, params):
    """Analyse the worst 10% out-of-fold predictions and propose new params."""
    err = per_row_error(task, y, oof)
    thr = np.quantile(err, 0.90)
    worst = err >= thr
    share = float(err[worst].sum() / max(err.sum(), 1e-12))

    # Which features differ most between worst-10% rows and the rest
    diffs = {}
    for c in X.columns:
        col = X[c].astype(float)
        sd = col.std()
        if sd and np.isfinite(sd):
            diffs[c] = float(abs(col[worst].mean() - col[~worst].mean()) / sd)
    top_shift = sorted(diffs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    max_shift = top_shift[0][1] if top_shift else 0.0

    new = dict(params)
    actions = []
    if share > 0.5:
        # Errors concentrated in few rows: likely outliers/noise -> regularise
        new["min_child_weight"] = params["min_child_weight"] * 3
        new["min_child_samples"] = params["min_child_samples"] * 2
        new["reg_lambda"] = params["reg_lambda"] * 3
        new["subsample"] = max(0.6, params["subsample"] - 0.1)
        actions.append("Errors concentrated (>50% of loss in worst 10%): increased regularisation")
    else:
        new["max_depth"] = params["max_depth"] + 2
        new["num_leaves"] = params["num_leaves"] * 2
        actions.append("Errors spread out: increased model capacity (depth/leaves)")
    if max_shift > 0.5:
        # Hard rows live in a distinct region of feature space -> upweight them
        actions.append(f"Worst rows form a distinct region (max shift {max_shift:.2f} SD): "
                       "upweighting hard examples x2")
        weights = np.where(worst, 2.0, 1.0)
    else:
        weights = None
    new["learning_rate"] = params["learning_rate"] * 0.7
    new["n_estimators"] = int(params["n_estimators"] * 1.4)
    actions.append("Lowered learning rate x0.7 and raised trees x1.4")
    summary = dict(worst10_error_share=round(share, 4),
                   top_shifted_features=[[k, round(v, 3)] for k, v in top_shift],
                   actions=actions)
    return new, weights, summary


def baseline_score(X, y, task, n_folds):
    import lightgbm as lgb
    folds = (KFold(n_folds, shuffle=True, random_state=SEED) if task == "regression"
             else StratifiedKFold(n_folds, shuffle=True, random_state=SEED))
    shape = (len(X),) if task != "multiclass" else (len(X), int(y.max()) + 1)
    oof = np.zeros(shape)
    for tr, va in folds.split(X, y):
        m = lgb.LGBMRegressor(verbose=-1, random_state=SEED) if task == "regression" else \
            lgb.LGBMClassifier(verbose=-1, random_state=SEED)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = predict(m, X.iloc[va], task)
    return score(task, y, oof)[0]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--state-file", default="state.json",
                    help="JSON with the best previous score (the baseline to beat)")
    ap.add_argument("--target", default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-train-rows", type=int, default=500_000,
                    help="Subsample training rows above this size to bound runtime")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    logs = []

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logs.append(line)

    train, test, sub, id_col, target, paths = load_data(args.data_dir, args.target, args.id_col)
    log(f"Loaded train {train.shape}, test {test.shape}; target='{target}', id='{id_col}'")
    if len(train) > args.max_train_rows:
        train = train.sample(args.max_train_rows, random_state=SEED).reset_index(drop=True)
        log(f"Subsampled train to {len(train)} rows")
    train = train.dropna(subset=[target]).reset_index(drop=True)

    task = infer_task(train[target])
    classes = None
    if task == "regression":
        y = train[target].astype(float).values
        n_classes = 0
    else:
        classes = sorted(train[target].unique().tolist(), key=str)
        mapping = {c: i for i, c in enumerate(classes)}
        y = train[target].map(mapping).astype(int).values
        n_classes = len(classes)
    log(f"Task: {task}" + (f" ({n_classes} classes)" if n_classes else ""))

    X, Xt, feats = engineer_features(train, test, target, id_col, task, y, log)

    # Baseline: previous best run, else a default single LightGBM
    state = {}
    if os.path.exists(args.state_file):
        with open(args.state_file) as f:
            state = json.load(f)
    _, higher_better, metric = score(task, y, np.zeros_like(y, dtype=float) if task != "multiclass"
                                     else np.full((len(y), n_classes), 1.0 / n_classes))
    if state.get("best_cv_score") is not None and state.get("metric") == metric:
        baseline = float(state["best_cv_score"])
        baseline_src = f"best previous run ({state.get('best_run_at', 'unknown')})"
    else:
        log("No previous baseline; computing default LightGBM baseline")
        baseline = baseline_score(X, y, task, args.folds)
        baseline_src = "default single LightGBM (first run)"
    log(f"Baseline {metric}: {baseline:.6f} [{baseline_src}]")

    # Round 1
    params = dict(n_estimators=400, learning_rate=0.05, max_depth=6, num_leaves=31,
                  subsample=0.8, colsample=0.8, min_child_weight=1, min_child_samples=20, reg_lambda=1.0)
    log("Round 1: stacking XGBoost + LightGBM + CatBoost")
    r1 = run_stack(X, y, Xt, task, n_classes, params, args.folds, None, log)
    log(f"Round 1 stack {metric}: {r1['stack_score']:.6f}  base: "
        + ", ".join(f"{k}={v:.5f}" for k, v in r1["base_scores"].items()))

    # Error analysis -> Round 2
    params2, weights, ea = error_analysis(X, y, r1["oof"], task, params)
    for a in ea["actions"]:
        log(f"Error analysis: {a}")
    log("Round 2: re-tuned stack")
    r2 = run_stack(X, y, Xt, task, n_classes, params2, args.folds, weights, log)
    log(f"Round 2 stack {metric}: {r2['stack_score']:.6f}")

    better = (lambda a, b: a > b) if higher_better else (lambda a, b: a < b)
    best_round, best = (2, r2) if better(r2["stack_score"], r1["stack_score"]) else (1, r1)
    final = best["stack_score"]
    beats = better(final, baseline)
    log(f"Selected round {best_round}: {metric}={final:.6f}; beats baseline: {beats}")

    # Submission
    sub_path = os.path.join(args.out_dir, "submission.csv")
    if beats:
        if task == "multiclass" and sub is not None and len(sub.columns) > 2:
            out = pd.DataFrame(best["test"], columns=[str(c) for c in classes])
            out.insert(0, id_col, test[id_col].values)
        else:
            pred = best["test"]
            if task == "multiclass":
                pred = np.array(classes)[pred.argmax(1)]
            col = sub.columns[1] if sub is not None and len(sub.columns) == 2 else target
            out = pd.DataFrame({col: pred})
            if id_col:
                out.insert(0, id_col, test[id_col].values)
        out.to_csv(sub_path, index=False)
        log(f"Wrote {sub_path} ({len(out)} rows)")
        state.update(best_cv_score=final, metric=metric, higher_is_better=higher_better,
                     best_run_at=datetime.now(timezone.utc).isoformat())
    else:
        if os.path.exists(sub_path):
            os.remove(sub_path)
        log("Did not beat baseline: no submission.csv written")
    state.setdefault("history", []).append(dict(
        run_at=datetime.now(timezone.utc).isoformat(), metric=metric, cv_score=final,
        baseline=baseline, beat_baseline=bool(beats)))
    with open(args.state_file, "w") as f:
        json.dump(state, f, indent=2)

    report = dict(
        version=VERSION, run_at=datetime.now(timezone.utc).isoformat(),
        task=task, metric=metric, higher_is_better=higher_better, target=target, id_col=id_col,
        features=feats, round1=dict(params=params, base_scores=r1["base_scores"], stack_score=r1["stack_score"]),
        error_analysis=ea,
        round2=dict(params=params2, base_scores=r2["base_scores"], stack_score=r2["stack_score"]),
        selected_round=best_round, final_cv_score=final, baseline=baseline, baseline_source=baseline_src,
        beat_baseline=bool(beats), submission_written=bool(beats), runtime_sec=round(time.time() - t0, 1),
        log=logs,
    )
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    md = [f"# Run report — {report['run_at']}", "",
          f"- Task: **{task}**, metric **{metric}** ({'higher' if higher_better else 'lower'} is better)",
          f"- Features: {feats['input_shape'][1]} → {feats['final_shape'][1]} "
          f"({len(feats['cross_features'])} cross features from top {feats['cross_feature_budget_k']} numeric columns)",
          f"- Round 1 stack: {r1['stack_score']:.6f}",
          f"- Error analysis: worst 10% carry {ea['worst10_error_share']:.1%} of the loss",
          *[f"  - {a}" for a in ea["actions"]],
          f"- Round 2 stack: {r2['stack_score']:.6f}",
          f"- **Final CV: {final:.6f}** vs baseline {baseline:.6f} ({baseline_src})",
          f"- Submission written: **{'yes' if beats else 'no'}**",
          f"- Runtime: {report['runtime_sec']}s"]
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    log("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
