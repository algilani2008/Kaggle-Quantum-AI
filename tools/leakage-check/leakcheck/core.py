"""Core checks for leakcheck.

Every check takes plain pandas DataFrames and returns a list of ``Finding``
objects. ``audit`` runs all checks that apply and returns a ``Report``.

Severity levels, from worst to mildest: ``high`` (very likely to inflate your
validation score), ``medium`` (worth a look), ``low`` (probably harmless but
noted), ``info`` (context only).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEVERITIES = ("high", "medium", "low", "info")
_RANK = {s: i for i, s in enumerate(SEVERITIES)}


@dataclass
class Thresholds:
    """Decision thresholds. Defaults were chosen to keep false alarms low on
    clean random splits (see benchmark.py and results/benchmark.md)."""

    # single-feature predictive power (AUC for classification, R^2 for regression)
    leak_auc_high: float = 0.995
    leak_auc_medium: float = 0.98
    # single-feature out-of-fold balanced accuracy (classification only); catches
    # near-copies of the label that have a few corrupted values
    leak_bacc_high: float = 0.99
    leak_bacc_medium: float = 0.96
    leak_r2_high: float = 0.98
    leak_r2_medium: float = 0.93
    # share of test rows that also appear (all features equal) in train
    dup_high: float = 0.05
    dup_medium: float = 0.005
    # adversarial validation AUC (train vs test)
    adv_high: float = 0.80
    adv_medium: float = 0.65
    # adversarial AUC must also beat 0.5 by this many standard errors (Hanley-McNeil),
    # so tiny test sets do not raise alarms from noise
    adv_min_z: float = 3.0
    # share of duplicated rows inside train (leaks across CV folds)
    train_dup_medium: float = 0.01


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class Report:
    findings: list
    summary: dict

    @property
    def worst(self) -> str:
        if not self.findings:
            return "info"
        return min((f.severity for f in self.findings), key=_RANK.__getitem__)

    def at_least(self, severity: str) -> list:
        return [f for f in self.findings if _RANK[f.severity] <= _RANK[severity]]

    def to_dict(self) -> dict:
        return {"summary": self.summary, "findings": [asdict(f) for f in self.findings]}

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), indent=2, default=_json_default, **kw)

    def to_markdown(self, title: str = "Leakage audit") -> str:
        lines = [f"# {title}", ""]
        s = self.summary
        lines.append(
            f"Train rows: {s.get('n_train')}, test rows: {s.get('n_test', '-')}, "
            f"features checked: {s.get('n_features')}, task: {s.get('task')}."
        )
        lines.append("")
        counts = {sev: len([f for f in self.findings if f.severity == sev]) for sev in SEVERITIES}
        lines.append("| Severity | Count |\n|---|---|")
        lines += [f"| {sev} | {counts[sev]} |" for sev in SEVERITIES]
        lines.append("")
        if not self.findings:
            lines.append("No findings.")
        for f in sorted(self.findings, key=lambda f: _RANK[f.severity]):
            lines.append(f"- **[{f.severity.upper()}] {f.check}**: {f.message}")
        lines.append("")
        return "\n".join(lines)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- helpers

def infer_task(y: pd.Series) -> str:
    """'classification' for non-numeric or few distinct values, else 'regression'."""
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    n_unique = y.nunique(dropna=True)
    if n_unique <= 20 and np.allclose(y.dropna() % 1, 0):
        return "classification"
    return "regression"


def _encode(col: pd.Series) -> np.ndarray:
    """Numeric columns pass through (NaN kept: sklearn trees handle it);
    other columns become integer codes with NaN for missing."""
    if pd.api.types.is_bool_dtype(col):
        return col.astype(float).to_numpy()
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(float).to_numpy()
    if pd.api.types.is_datetime64_any_dtype(col):
        out = col.astype("int64").astype(float).to_numpy()
        out[col.isna().to_numpy()] = np.nan
        return out
    codes, _ = pd.factorize(col.astype("string"), sort=True)
    codes = codes.astype(float)
    codes[codes < 0] = np.nan
    return codes


def _row_hashes(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df.astype("string").fillna("<NA>"), index=False)


def _univariate_score(x: np.ndarray, y: np.ndarray, task: str, seed: int, folds: int = 3) -> float:
    """Out-of-fold score of a small tree that sees only one column.

    AUC (macro one-vs-rest for multiclass) for classification, R^2 for regression.
    A shallow tree catches monotonic and non-monotonic relations, and lookups
    of categorical codes, without overfitting much.
    """
    return _univariate_scores(x, y, task, seed, folds)["score"]


def _univariate_scores(x: np.ndarray, y: np.ndarray, task: str, seed: int, folds: int = 3) -> dict:
    """Like _univariate_score but also returns balanced accuracy for classification."""
    X = x.reshape(-1, 1)
    if task == "classification":
        classes, y_enc = np.unique(y, return_inverse=True)
        if len(classes) < 2 or np.bincount(y_enc).min() < folds:
            return {"score": float("nan"), "bacc": float("nan")}
        model = DecisionTreeClassifier(max_leaf_nodes=32, min_samples_leaf=5, random_state=seed)
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        proba = cross_val_predict(model, X, y_enc, cv=cv, method="predict_proba")
        bacc = float(balanced_accuracy_score(y_enc, proba.argmax(1)))
        if len(classes) == 2:
            auc = float(roc_auc_score(y_enc, proba[:, 1]))
        else:
            auc = float(roc_auc_score(y_enc, proba, multi_class="ovr", average="macro"))
        return {"score": auc, "bacc": bacc}
    model = DecisionTreeRegressor(max_leaf_nodes=32, min_samples_leaf=5, random_state=seed)
    cv = KFold(folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, y.astype(float), cv=cv)
    return {"score": float(r2_score(y.astype(float), pred)), "bacc": float("nan")}


# --------------------------------------------------------------------------- checks

def check_cross_duplicates(train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str],
                           th: Thresholds) -> list:
    """Test rows whose feature values all appear in a train row."""
    h_train = set(_row_hashes(train[features]))
    in_train = _row_hashes(test[features]).isin(h_train)
    share = float(in_train.mean()) if len(test) else 0.0
    if share == 0:
        return []
    uniq_ratio = float(train[features].drop_duplicates().shape[0] / max(len(train), 1))
    sev = "high" if share >= th.dup_high else "medium" if share >= th.dup_medium else "low"
    note = ""
    if uniq_ratio < 0.5:
        # Few distinct rows exist at all (e.g. 3 categorical columns): repeats are natural.
        sev = "low" if sev != "high" else "medium"
        note = (f" Only {uniq_ratio:.0%} of train rows are distinct, so repeats are expected "
                "for this data; this is a weaker signal.")
    return [Finding(
        "cross_duplicates", sev,
        f"{int(in_train.sum())} of {len(test)} test rows ({share:.1%}) have exact copies in train."
        + note,
        {"n": int(in_train.sum()), "share": share, "train_unique_ratio": uniq_ratio,
         "example_test_index": [str(i) for i in test.index[in_train.to_numpy()][:5]]},
    )]


def check_train_duplicates(train: pd.DataFrame, features: Sequence[str], target: Optional[str],
                           th: Thresholds) -> list:
    """Duplicated rows inside train leak across CV folds; conflicting labels signal noise."""
    out = []
    dup_mask = train.duplicated(subset=list(features), keep=False)
    share = float(train.duplicated(subset=list(features)).mean()) if len(train) else 0.0
    if share > 0:
        uniq_ratio = 1 - share
        sev = "medium" if share >= th.train_dup_medium and uniq_ratio >= 0.5 else "low"
        if uniq_ratio >= 0.5:
            msg = (f"{share:.1%} of train rows repeat an earlier row on all features; with plain "
                   "KFold the copies land in different folds and inflate CV scores (use GroupKFold "
                   "on a row hash).")
        else:
            msg = (f"{share:.1%} of train rows repeat an earlier row on all features. With so few "
                   "distinct rows this is probably natural to the data, not a pipeline bug.")
        out.append(Finding("train_duplicates", sev, msg, {"share": share}))
    if target is not None and dup_mask.any():
        g = train.loc[dup_mask].groupby(list(features), dropna=False)[target].nunique()
        n_conf = int((g > 1).sum())
        if n_conf:
            out.append(Finding(
                "conflicting_labels", "info",
                f"{n_conf} groups of identical feature rows carry different targets "
                "(label noise or missing features).",
                {"groups": n_conf},
            ))
    return out


def check_id_like(train: pd.DataFrame, features: Sequence[str], target: Optional[str], task: str,
                  th: Thresholds, seed: int) -> list:
    """Columns that are unique for (almost) every row, like IDs, row numbers or timestamps."""
    out = []
    n = len(train)
    for c in features:
        col = train[c]
        if col.nunique(dropna=True) < 0.95 * n:
            continue
        is_float = pd.api.types.is_float_dtype(col) and not np.allclose(col.dropna() % 1, 0)
        if is_float:
            continue  # continuous measurements are naturally unique
        sev, extra = "low", ""
        details = {}
        if target is not None:
            score = _univariate_score(_encode(col), train[target].to_numpy(), task, seed)
            details["target_score"] = score
            strong = (score >= th.leak_auc_medium - 0.08) if task == "classification" else (score >= 0.5)
            if strong:
                sev = "high"
                extra = (f" It also predicts the target on its own (score {score:.3f}), which suggests "
                         "rows were ordered by target or the ID encodes the label.")
            else:
                sev = "medium" if score > (0.6 if task == "classification" else 0.1) else "low"
                extra = f" Single-column score against the target: {score:.3f}."
        out.append(Finding("id_like_column", sev,
                           f"Column '{c}' is unique per row (looks like an ID). Drop it before training."
                           + extra, details))
    return out


def check_target_leakage(train: pd.DataFrame, features: Sequence[str], target: str, task: str,
                         th: Thresholds, seed: int, skip: Iterable[str] = ()) -> tuple:
    """Single columns that predict the target almost perfectly on their own."""
    y = train[target].to_numpy()
    scores = {}
    out = []
    skip = set(skip)
    for c in features:
        r = _univariate_scores(_encode(train[c]), y, task, seed)
        s, bacc = r["score"], r["bacc"]
        scores[c] = s
        if c in skip or np.isnan(s):
            continue
        if task == "classification":
            metric = "AUC"
            high = s >= th.leak_auc_high or bacc >= th.leak_bacc_high
            med = s >= th.leak_auc_medium or bacc >= th.leak_bacc_medium
            shown = f"AUC {s:.3f}, balanced accuracy {bacc:.3f}"
        else:
            metric = "R^2"
            high, med = s >= th.leak_r2_high, s >= th.leak_r2_medium
            shown = f"R^2 {s:.3f}"
        if med:
            sev = "high" if high else "medium"
            out.append(Finding("target_leakage", sev,
                               f"Column '{c}' alone predicts the target out-of-fold ({shown}). "
                               "Check that it is known at prediction time and is not computed "
                               "from the target.",
                               {"column": c, "score": s, "balanced_accuracy": bacc, "metric": metric}))
    return out, scores


def check_adversarial(train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str],
                      th: Thresholds, seed: int, top_k: int = 5) -> tuple:
    """Adversarial validation: can a model tell train rows from test rows?"""
    both = pd.concat([train[features], test[features]], axis=0, ignore_index=True)
    X = np.column_stack([_encode(both[c]) for c in features])
    is_test = np.r_[np.zeros(len(train)), np.ones(len(test))].astype(int)
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed)
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X, is_test, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(is_test, proba))
    per_feature = {c: _univariate_score(X[:, i], is_test, "classification", seed)
                   for i, c in enumerate(features)}
    top = sorted(per_feature.items(), key=lambda kv: -np.nan_to_num(kv[1]))[:top_k]
    # Hanley & McNeil (1982) standard error of an AUC near 0.5 for these class sizes
    n1, n0 = len(test), len(train)
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    se = float(np.sqrt(max(auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2)
                           + (n0 - 1) * (q2 - auc ** 2), 1e-12) / (n1 * n0)))
    z = (auc - 0.5) / se
    sev = "high" if auc >= th.adv_high else "medium" if auc >= th.adv_medium else "info"
    if sev != "info" and z < th.adv_min_z:
        sev = "low"  # could be noise: too few rows to be sure
    tops = ", ".join(f"{c} ({s:.2f})" for c, s in top)
    msg = (f"A model separates train from test rows with AUC {auc:.3f} "
           f"(0.5 = indistinguishable, z = {z:.1f}). Most shifted columns: {tops}.")
    if sev != "info":
        msg += " Your validation split may not look like the test set; validate on rows that resemble test."
    return [Finding("adversarial_validation", sev, msg,
                    {"auc": auc, "z": z, "top_features": dict(top)})], auc


def check_group_overlap(train: pd.DataFrame, test: pd.DataFrame, group: str) -> list:
    shared = set(train[group].dropna()) & set(test[group].dropna())
    if not shared:
        return []
    share = float(test[group].isin(shared).mean())
    return [Finding("group_overlap", "high" if share >= 0.05 else "medium",
                    f"{len(shared)} values of '{group}' appear in both train and test, covering "
                    f"{share:.1%} of test rows. If the model will meet new {group}s, split by group.",
                    {"shared_groups": len(shared), "test_share": share})]


def check_time_order(train: pd.DataFrame, test: pd.DataFrame, time: str) -> list:
    t_train = pd.to_datetime(train[time], errors="coerce")
    t_test = pd.to_datetime(test[time], errors="coerce")
    if t_train.isna().all() or t_test.isna().all():
        return []
    share = float((t_test <= t_train.max()).mean())
    if share == 0:
        return []
    return [Finding("time_overlap", "high" if share >= 0.05 else "medium",
                    f"{share:.1%} of test rows are not later than the newest train row "
                    f"({t_train.max()}). For forecasting, train only on the past.",
                    {"test_share_not_after_train": share})]


# --------------------------------------------------------------------------- main entry

def audit(train: pd.DataFrame, test: Optional[pd.DataFrame] = None, target: Optional[str] = None,
          task: str = "auto", group: Optional[str] = None, time: Optional[str] = None,
          ignore: Sequence[str] = (), thresholds: Optional[Thresholds] = None,
          max_rows: int = 20000, random_state: int = 0) -> Report:
    """Run every applicable leakage check and return a Report.

    train, test : DataFrames. ``test`` may be None (then only train-side checks run)
                  and may lack the target column.
    target      : name of the label column in train.
    task        : 'classification', 'regression' or 'auto'.
    group, time : optional group-id and timestamp columns for split checks.
    ignore      : columns to leave out of every check.
    max_rows    : train/test are randomly subsampled to at most this many rows for speed.
    """
    th = thresholds or Thresholds()
    if target is not None and target not in train.columns:
        raise ValueError(f"target column '{target}' not in train")
    rng = np.random.RandomState(random_state)

    def _sample(df):
        if df is None or len(df) <= max_rows:
            return df
        return df.iloc[np.sort(rng.choice(len(df), max_rows, replace=False))]

    train_s, test_s = _sample(train), _sample(test)
    if target is not None:
        train_s = train_s[train_s[target].notna()]
    excluded = set(ignore) | {target, group, time} - {None}
    features = [c for c in train.columns if c not in excluded]
    if test is not None:
        features = [c for c in features if c in test.columns]
    if not features:
        raise ValueError("no feature columns left to check")

    task_ = infer_task(train_s[target]) if (target is not None and task == "auto") else task
    findings: list = []
    summary = {"n_train": len(train), "n_test": None if test is None else len(test),
               "n_features": len(features), "task": task_ if target is not None else None}

    findings += check_train_duplicates(train_s, features, target, th)
    id_findings = check_id_like(train_s, features, target, task_, th, random_state)
    findings += id_findings
    if target is not None:
        id_cols = {f.message.split("'")[1] for f in id_findings}
        leak, scores = check_target_leakage(train_s, features, target, task_, th, random_state,
                                            skip=id_cols)
        findings += leak
        summary["univariate_scores"] = scores
    if test is not None:
        findings += check_cross_duplicates(train_s, test_s, features, th)
        adv, auc = check_adversarial(train_s, test_s, features, th, random_state)
        findings += adv
        summary["adversarial_auc"] = auc
        if group is not None:
            findings += check_group_overlap(train_s, test_s, group)
        if time is not None:
            findings += check_time_order(train_s, test_s, time)
    findings.sort(key=lambda f: _RANK[f.severity])
    return Report(findings, summary)
