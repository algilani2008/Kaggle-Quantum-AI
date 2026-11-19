"""Benchmark: do sigdiff's tests behave as promised on real data?

Data: UCI "Adult" census income (48,842 rows, CC BY 4.0), downloaded from a
public GitHub mirror at run time (not committed).

Three experiments, all with real trained models:

1. Null check (false alarms). For each simulated test set we randomly swap the
   predictions of two real models row by row. The two resulting "models" are
   exactly equally good in expectation, so any test that says "different"
   is a false alarm. A good 5%-level test should be wrong ~5% of the time.
2. Power and coverage. Real model pairs with a real gap (logistic regression vs
   gradient boosting, and a small gap between two boosting configs). For test
   sets of different sizes we measure how often each test detects the gap and
   how often the 95% bootstrap interval contains the true gap measured on the
   ~30k-row pool.
3. Cross-validation. Compare two algorithms with 10x10-fold CV on 300-row
   datasets and check how often each CV test's 95% interval contains the true
   expected difference (estimated from 400 independent training sets).

Run:  python benchmark/run_benchmark.py            (about 10-20 min on 2 CPUs)
      python benchmark/run_benchmark.py --quick    (a smoke test, ~1 min)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import sigdiff as sd  # noqa: E402

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
        "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
# Protected attributes are dropped: this benchmark is about statistics, not about profiling people.
DROP = ["fnlwgt", "race", "sex", "native_country"]
CACHE = Path(os.environ.get("SIGDIFF_CACHE", Path.home() / ".cache" / "sigdiff"))


def load_adult():
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "adult-all.csv"
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=60) as r:
            path.write_bytes(r.read())
    df = pd.read_csv(path, header=None, names=COLS, na_values="?", skipinitialspace=True)
    df = df.dropna(subset=["income"]).reset_index(drop=True)
    y = (df["income"].str.strip().str.startswith(">50K")).astype(int).to_numpy()
    X = df.drop(columns=DROP + ["income"])
    cat = X.select_dtypes(include=["object", "string"]).columns
    X = pd.get_dummies(X.fillna({c: "missing" for c in cat}), columns=list(cat), dtype=float)
    return X.to_numpy(dtype=float), y


def unpaired_bootstrap_p(y, a, b, metric, n_resamples, rng):
    """Bootstrap that (wrongly) resamples A's and B's test rows independently."""
    m = sd.get_metric(metric)
    n = y.size
    d = np.empty(n_resamples)
    for r in range(n_resamples):
        ia, ib = rng.integers(0, n, n), rng.integers(0, n, n)
        d[r] = m.score(y[ib], b[ib]) - m.score(y[ia], a[ia])
    return min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))


def run_tests(y, a, b, metric, n_res, seed):
    """All holdout tests on one test set -> dict of p-values plus bootstrap CI."""
    rng = np.random.default_rng(seed)
    boot = sd.paired_bootstrap(y, a, b, metric, n_resamples=n_res, seed=seed)
    out = {
        "diff": boot.diff, "ci_low": boot.ci_low, "ci_high": boot.ci_high,
        "paired_bootstrap": boot.p_value,
        "permutation": sd.permutation_test(y, a, b, metric, n_resamples=n_res, seed=seed + 1).p_value,
        "unpaired_bootstrap": unpaired_bootstrap_p(y, a, b, metric, n_res, rng),
    }
    if metric == "accuracy":
        out["mcnemar"] = sd.mcnemar(y, a, b).p_value
    return out


# ------------------------------------------------------------------ experiment 1
def null_sim(s, y, pa, pb, n, metrics, n_res):
    rng = np.random.default_rng(10_000 + s)
    idx = rng.choice(y.size, n, replace=False)
    swap = rng.random(n) < 0.5
    a = np.where(swap, pb[idx], pa[idx])
    b = np.where(swap, pa[idx], pb[idx])
    rows = []
    for m in metrics:
        rows.append({"sim": s, "metric": m, "n": n, **run_tests(y[idx], a, b, m, n_res, s)})
    return rows


def noise_floor(y, pa, pb, ns, sims, seed=7):
    """95th percentile of |accuracy gap| between two equally good models, by test size."""
    rng = np.random.default_rng(seed)
    corr_a, corr_b = (pa >= 0.5) == y, (pb >= 0.5) == y
    out = []
    for n in ns:
        gaps = np.empty(sims)
        for s in range(sims):
            idx = rng.choice(y.size, n, replace=False)
            swap = rng.random(n) < 0.5
            ca = np.where(swap, corr_b[idx], corr_a[idx])
            cb = np.where(swap, corr_a[idx], corr_b[idx])
            gaps[s] = cb.mean() - ca.mean()
        out.append({"n": n, "abs_gap_p95_pct_points": 100 * float(np.quantile(np.abs(gaps), 0.95))})
    return out


# ------------------------------------------------------------------ experiment 2
def power_sim(s, y, pa, pb, n, metrics, n_res):
    rng = np.random.default_rng(20_000 + s * 7 + n)
    idx = rng.choice(y.size, n, replace=False)
    return [{"sim": s, "metric": m, "n": n, **run_tests(y[idx], pa[idx], pb[idx], m, n_res, s)} for m in metrics]


# ------------------------------------------------------------------ experiment 3
def make_algos():
    return (lambda: LogisticRegression(max_iter=2000, C=1.0),
            lambda: DecisionTreeClassifier(max_depth=4, random_state=0))


def fit_score(make, Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    model = make().fit(sc.transform(Xtr), ytr)
    return float(np.mean(model.predict(sc.transform(Xte)) == yte))


def true_cv_gap(X, y, holdout, n_train, reps, seed):
    """Expected accuracy(tree) - accuracy(logreg) when trained on n_train rows."""
    rng = np.random.default_rng(seed)
    la, lb = make_algos()
    pool = np.setdiff1d(np.arange(y.size), holdout)
    gaps = []
    for _ in range(reps):
        tr = rng.choice(pool, n_train, replace=False)
        if len(np.unique(y[tr])) < 2:
            continue
        gaps.append(fit_score(lb, X[tr], y[tr], X[holdout], y[holdout]) -
                    fit_score(la, X[tr], y[tr], X[holdout], y[holdout]))
    return float(np.mean(gaps)), float(np.std(gaps, ddof=1) / np.sqrt(len(gaps)))


def cv_sim(s, X, y, pool, N):
    rng = np.random.default_rng(30_000 + s)
    idx = rng.choice(pool, N, replace=False)
    Xs, ys = X[idx], y[idx]
    la, lb = make_algos()
    sa, sb = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=s).split(Xs, ys):
        sa.append(fit_score(la, Xs[tr], ys[tr], Xs[te], ys[te]))
        sb.append(fit_score(lb, Xs[tr], ys[tr], Xs[te], ys[te]))
    a52, b52 = np.empty((5, 2)), np.empty((5, 2))
    for r in range(5):
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=1000 * s + r).split(Xs, ys)
        for f, (tr, te) in enumerate(folds):
            a52[r, f] = fit_score(la, Xs[tr], ys[tr], Xs[te], ys[te])
            b52[r, f] = fit_score(lb, Xs[tr], ys[tr], Xs[te], ys[te])
    n_test = N // 10
    res = {
        "naive_ttest": sd.naive_ttest(sa, sb),
        "corrected_resampled_ttest": sd.corrected_resampled_ttest(sa, sb, N - n_test, n_test),
        "five_by_two_cv_ttest": sd.five_by_two_cv_ttest(a52, b52),
    }
    return [{"sim": s, "test": k, "diff": r.diff, "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p_value}
            for k, r in res.items()]


# ------------------------------------------------------------------ charts
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, AXIS = "#1f1f1d", "#6b6a64", "#e1e0d9", "#c3c2b7"
METHOD_COLORS = {"paired_bootstrap": BLUE, "permutation": ORANGE, "mcnemar": AQUA, "unpaired_bootstrap": YELLOW}
LABEL = {"paired_bootstrap": "Paired bootstrap", "permutation": "Paired permutation", "mcnemar": "McNemar",
         "unpaired_bootstrap": "Unpaired bootstrap (wrong)", "naive_ttest": "Naive t-test (wrong)",
         "corrected_resampled_ttest": "Corrected t-test", "five_by_two_cv_ttest": "5x2cv t-test"}


def _style(ax):
    import matplotlib.pyplot as plt  # noqa: F401
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(null_df, power_df, cv_df, out):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. false alarm rate under the null
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, m in zip(axes, ["accuracy", "auc", "logloss"]):
        d = null_df[null_df.metric == m]
        meths = [k for k in METHOD_COLORS if k in d and d[k].notna().any()]
        rates = [100 * np.mean(d[k] < 0.05) for k in meths]
        ax.bar(range(len(meths)), rates, color=[METHOD_COLORS[k] for k in meths], width=0.6)
        ax.axhline(5, color=INK, linewidth=1, linestyle="--")
        for i, r in enumerate(rates):
            ax.text(i, r / 2 if r > 1 else r + 0.3, f"{r:.1f}%", ha="center", fontsize=8,
                    color="#ffffff" if r > 1 else INK)
        ax.set_xticks(range(len(meths)))
        short = {"paired_bootstrap": "Paired\nbootstrap", "permutation": "Paired\npermutation",
                 "mcnemar": "McNemar", "unpaired_bootstrap": "Unpaired\n(wrong)"}
        ax.set_xticklabels([short[k] for k in meths], fontsize=8)
        ax.set_title(m, fontsize=10, color=INK)
        _style(ax)
    axes[0].set_ylabel("False alarms (%)", color=MUTED)
    fig.suptitle("Two equally good models, 1,000 test rows: how often does each test cry 'different'?\n"
                 "dashed line = the promised 5%", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "null_false_alarms.png", dpi=150)
    plt.close(fig)

    # 2. power vs test size, per pair, accuracy + auc
    pairs = power_df.pair.unique()
    fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 3.4 * len(pairs)), squeeze=False)
    for i, pair in enumerate(pairs):
        for j, m in enumerate(["accuracy", "auc"]):
            ax = axes[i, j]
            d = power_df[(power_df.pair == pair) & (power_df.metric == m)]
            for k, col in METHOD_COLORS.items():
                if k not in d or d[k].isna().all():
                    continue
                g = d.groupby("n")[k].apply(lambda p: 100 * np.mean(p < 0.05))
                ax.plot(g.index, g.values, marker="o", markersize=5, linewidth=2, color=col, label=LABEL[k])
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks(sorted(d.n.unique()))
            ax.set_xticklabels([str(v) for v in sorted(d.n.unique())])
            ax.set_ylim(0, 102)
            ax.set_title(f"{pair} - {m}", fontsize=10, color=INK)
            ax.set_xlabel("test rows", color=MUTED)
            ax.set_ylabel("gap detected (%)", color=MUTED)
            _style(ax)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "power_by_test_size.png", dpi=150)
    plt.close(fig)

    # 3. CV interval coverage
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    tests = ["naive_ttest", "corrected_resampled_ttest", "five_by_two_cv_ttest"]
    cov = [100 * cv_df[cv_df.test == t].covered.mean() for t in tests]
    ax.bar(range(3), cov, color=[YELLOW, BLUE, ORANGE], width=0.55)
    ax.axhline(95, color=INK, linestyle="--", linewidth=1)
    for i, c in enumerate(cov):
        ax.text(i, c / 2, f"{c:.0f}%", ha="center", fontsize=10, color="#ffffff")
    ax.set_xticks(range(3))
    ax.set_xticklabels([LABEL[t] for t in tests], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("95% intervals that contain\nthe true gap (%)", color=MUTED)
    ax.set_title("Comparing algorithms with cross-validation (300 rows, 10x10 CV)", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "cv_interval_coverage.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    q = args.quick
    cfg = dict(
        null_sims=60 if q else 1000, null_n=1000, n_res=300 if q else 1000,
        power_sims=30 if q else 300, power_ns=[250, 1000] if q else [250, 500, 1000, 2000, 4000],
        power_res=200 if q else 500, cv_sims=12 if q else 300, cv_N=300, true_reps=40 if q else 400,
        floor_sims=500 if q else 5000,
    )

    X, y = load_adult()
    rng = np.random.default_rng(0)
    perm = rng.permutation(y.size)
    train, cv_holdout, pool = perm[:8000], perm[8000:18000], perm[18000:]
    print(f"Adult: {y.size} rows, {X.shape[1]} features, positive rate {y.mean():.3f}; "
          f"train {train.size}, CV holdout {cv_holdout.size}, evaluation pool {pool.size}")

    sc = StandardScaler().fit(X[train])
    models = {
        "logreg": LogisticRegression(max_iter=3000),
        "tree": DecisionTreeClassifier(max_depth=3, random_state=0),
        "gbm": HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=8, learning_rate=0.1, random_state=0),
    }
    probs = {}
    for k, mdl in models.items():
        mdl.fit(sc.transform(X[train]), y[train])
        probs[k] = mdl.predict_proba(sc.transform(X[pool]))[:, 1]
    yp = y[pool]
    pool_scores = {k: {m: sd.get_metric(m).score(yp, p) for m in ["accuracy", "auc", "logloss"]} for k, p in probs.items()}
    print("pool scores:", json.dumps(pool_scores, indent=1))

    metrics = ["accuracy", "auc", "logloss"]
    par = Parallel(n_jobs=args.jobs)

    # 1
    rows = par(delayed(null_sim)(s, yp, probs["logreg"], probs["gbm"], cfg["null_n"], metrics, cfg["n_res"])
               for s in range(cfg["null_sims"]))
    null_df = pd.DataFrame([r for rr in rows for r in rr])
    floors = noise_floor(yp, probs["logreg"], probs["gbm"], [250, 500, 1000, 2000, 4000, 10000], cfg["floor_sims"])
    print(f"exp1 done {time.time() - t0:.0f}s")

    # 2
    pairs = {"tree vs gbm (big gap)": ("tree", "gbm"), "logreg vs gbm (small gap)": ("logreg", "gbm")}
    pdfs = []
    for name, (ka, kb) in pairs.items():
        jobs = [(s, n) for n in cfg["power_ns"] for s in range(cfg["power_sims"])]
        rows = par(delayed(power_sim)(s, yp, probs[ka], probs[kb], n, metrics, cfg["power_res"]) for s, n in jobs)
        d = pd.DataFrame([r for rr in rows for r in rr])
        d["pair"] = name
        d["true_diff"] = d.metric.map({m: pool_scores[kb][m] - pool_scores[ka][m] for m in metrics})
        pdfs.append(d)
    power_df = pd.concat(pdfs, ignore_index=True)
    power_df["covered"] = (power_df.ci_low <= power_df.true_diff) & (power_df.true_diff <= power_df.ci_high)
    print(f"exp2 done {time.time() - t0:.0f}s")

    # 3
    N = cfg["cv_N"]
    true_10, se_10 = true_cv_gap(X, y, cv_holdout, N - N // 10, cfg["true_reps"], 1)
    true_2, se_2 = true_cv_gap(X, y, cv_holdout, N // 2, cfg["true_reps"], 2)
    rows = par(delayed(cv_sim)(s, X, y, pool, N) for s in range(cfg["cv_sims"]))
    cv_df = pd.DataFrame([r for rr in rows for r in rr])
    cv_df["true_diff"] = np.where(cv_df.test == "five_by_two_cv_ttest", true_2, true_10)
    cv_df["covered"] = (cv_df.ci_low <= cv_df.true_diff) & (cv_df.true_diff <= cv_df.ci_high)
    print(f"exp3 done {time.time() - t0:.0f}s")

    # ---- summaries
    def rate(s):
        return round(100 * float(np.mean(s < 0.05)), 1)

    null_summary = {m: {k: rate(null_df[null_df.metric == m][k]) for k in METHOD_COLORS
                        if k in null_df and null_df[null_df.metric == m][k].notna().any()} for m in metrics}
    power_summary = (power_df.groupby(["pair", "metric", "n"])
                     .agg(true_diff=("true_diff", "first"),
                          paired_bootstrap=("paired_bootstrap", rate), permutation=("permutation", rate),
                          unpaired_bootstrap=("unpaired_bootstrap", rate),
                          mcnemar=("mcnemar", lambda s: rate(s) if s.notna().any() else np.nan),
                          ci_coverage=("covered", lambda s: round(100 * s.mean(), 1)))
                     .reset_index())
    cv_summary = (cv_df.groupby("test").agg(coverage=("covered", lambda s: round(100 * s.mean(), 1)),
                                            significant=("p", rate), mean_diff=("diff", "mean"),
                                            mean_ci_width=("ci_high", "mean")).reset_index())
    cv_summary["mean_ci_width"] = (cv_df.ci_high - cv_df.ci_low).groupby(cv_df.test).mean().values

    power_summary.to_csv(out / "power_and_coverage.csv", index=False)
    cv_summary.to_csv(out / "cv_tests.csv", index=False)
    pd.DataFrame(floors).to_csv(out / "noise_floor_accuracy.csv", index=False)
    summary = {
        "config": cfg,
        "data": {"rows": int(y.size), "features": int(X.shape[1]), "positive_rate": round(float(y.mean()), 4),
                 "train": int(train.size), "pool": int(pool.size)},
        "pool_scores": pool_scores,
        "null_false_alarm_pct_at_alpha_0.05": null_summary,
        "null_false_alarm_binomial_95pct_halfwidth": round(100 * 1.96 * np.sqrt(0.05 * 0.95 / cfg["null_sims"]), 2),
        "noise_floor_accuracy": floors,
        "cv": {"N": N, "true_gap_10fold_train": [true_10, se_10], "true_gap_2fold_train": [true_2, se_2],
               "tests": cv_summary.to_dict(orient="records")},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    null_df.to_csv(out / "raw_null_sims.csv.gz", index=False, float_format="%.6g")
    power_df.to_csv(out / "raw_power_sims.csv.gz", index=False, float_format="%.6g")
    cv_df.to_csv(out / "raw_cv_sims.csv.gz", index=False, float_format="%.6g")
    charts(null_df, power_df, cv_df, out)
    print(json.dumps({k: summary[k] for k in ["null_false_alarm_pct_at_alpha_0.05", "noise_floor_accuracy", "cv"]}, indent=1))
    print(power_summary.to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
