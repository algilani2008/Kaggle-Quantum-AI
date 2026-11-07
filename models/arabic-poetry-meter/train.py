"""Train and evaluate the Arabic poetry meter classifier on MetRec.

Steps
1. Download MetRec (checksum-verified) into ./cache.
2. Hold out a stratified 10% of the official train split as validation.
3. Compare baselines and feature ablations on validation; pick C on validation.
   Then test whether adding diacritic-free copies of the training verses helps.
4. Retrain the chosen model on the full train split; evaluate ONCE on the
   official test split (and on a de-duplicated copy without verses that also
   appear in train).
5. Save the model (JSON + .npz, no pickle) into ./weights and check that the
   saved model reproduces the in-memory predictions.

Usage:  python train.py            (about 10-15 minutes on 2 CPU cores)
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from bahr.data import download, read_split
from bahr.features import diacritic_density, strip_diacritics
from bahr.labels import METERS
from bahr.metrics import accuracy, bootstrap_ci, confusion, macro_f1, per_class_report
from bahr.model import MeterClassifier

HERE = Path(__file__).resolve().parent
SEED = 0


def majority_baseline(y_train, n):
    values, counts = np.unique(y_train, return_counts=True)
    return np.full(n, values[np.argmax(counts)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(HERE / "cache"))
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--weights", default=str(HERE / "weights"))
    ap.add_argument("--c-grid", default="0.25,0.5,1.0")
    args = ap.parse_args()
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    folder = download(args.cache)
    X_train_full, y_train_full = read_split(folder, "train")
    X_test, y_test = read_split(folder, "test")
    y_train_full, y_test = np.array(y_train_full), np.array(y_test)
    print(f"train {len(X_train_full)}  test {len(X_test)}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=SEED
    )

    # ---------- 1. model selection on validation only ----------
    val_rows = []

    def record(name, pred):
        row = {"model": name, "val_accuracy": accuracy(y_val, pred), "val_macro_f1": macro_f1(y_val, pred)}
        val_rows.append(row)
        print(f"  [val] {name:38s} acc={row['val_accuracy']:.4f}  macroF1={row['val_macro_f1']:.4f}", flush=True)

    record("majority class", majority_baseline(y_tr, len(y_val)))
    ablations = [
        ("length only", ("length",)),
        ("characters only (bag of char n-grams)", ("text",)),
        ("rhythm skeleton only", ("rhythm",)),
        ("characters + rhythm", ("text", "rhythm")),
    ]
    for name, blocks in ablations:
        record(name, MeterClassifier(blocks=blocks, C=0.5).fit(X_tr, y_tr).predict(X_val))

    best_c, best_acc = None, -1.0
    for c in [float(x) for x in args.c_grid.split(",")]:
        pred = MeterClassifier(C=c).fit(X_tr, y_tr).predict(X_val)
        record(f"full (chars + rhythm + length), C={c}", pred)
        if val_rows[-1]["val_accuracy"] > best_acc:
            best_c, best_acc = c, val_rows[-1]["val_accuracy"]
    print(f"chosen C = {best_c}")

    # Augmentation: also train on a diacritic-free copy of every verse, so the
    # model still works when people type verses without tashkeel.
    def augment(X, y):
        return list(X) + [strip_diacritics(v) for v in X], np.concatenate([y, y])

    X_val_bare = [strip_diacritics(v) for v in X_val]
    plain = MeterClassifier(C=best_c).fit(X_tr, y_tr)
    aug = MeterClassifier(C=best_c).fit(*augment(X_tr, y_tr))
    record(f"full, C={best_c}, + diacritic-free copies", aug.predict(X_val))
    val_bare = {
        "full_plain": accuracy(y_val, plain.predict(X_val_bare)),
        "full_augmented": accuracy(y_val, aug.predict(X_val_bare)),
    }
    use_aug = val_rows[-1]["val_accuracy"] >= best_acc
    print(f"  [val, diacritics removed] plain={val_bare['full_plain']:.4f} augmented={val_bare['full_augmented']:.4f}; use augmentation: {use_aug}")

    # ---------- 2. final model, evaluated once on the official test split ----------
    X_fit, y_fit = augment(X_train_full, y_train_full) if use_aug else (X_train_full, y_train_full)
    model = MeterClassifier(C=best_c).fit(X_fit, y_fit)
    pred = model.predict(X_test)

    train_set = set(X_train_full)
    in_train = np.array([v in train_set for v in X_test])
    keep = ~in_train
    lo, hi = bootstrap_ci(y_test, pred, n_boot=1000, seed=SEED)

    # baselines refit on full train for the test table
    test_rows = [{"model": "majority class", "test_accuracy": accuracy(y_test, majority_baseline(y_train_full, len(y_test)))}]
    for name, blocks in [("length only", ("length",)), ("characters only (bag of char n-grams)", ("text",))]:
        p = MeterClassifier(blocks=blocks, C=0.5).fit(X_train_full, y_train_full).predict(X_test)
        test_rows.append({"model": name, "test_accuracy": accuracy(y_test, p)})
    test_rows.append({"model": f"bahr full model (C={best_c}{', augmented' if use_aug else ''})", "test_accuracy": accuracy(y_test, pred)})

    # robustness: same verses with every diacritic removed
    pred_bare = model.predict([strip_diacritics(v) for v in X_test])

    # accuracy by how much of the verse is diacritized
    dens = np.array([diacritic_density(v) for v in X_test])
    bins = [(0.0, 0.05, "none (<5%)"), (0.05, 0.4, "partial (5-40%)"), (0.4, 1.01, "heavy (>=40%)")]
    by_diac = []
    for a, b, name in bins:
        m = (dens >= a) & (dens < b)
        by_diac.append({"diacritics": name, "n": int(m.sum()), "accuracy": accuracy(y_test[m], pred[m]) if m.any() else None})

    # ---------- 3. save, reload, verify ----------
    model.save(args.weights, half_precision=True)
    reloaded = MeterClassifier.load(args.weights)
    pred_reloaded = reloaded.predict(X_test)
    agreement = float(np.mean(pred_reloaded == pred))
    size_mb = sum(p.stat().st_size for p in Path(args.weights).iterdir()) / 1e6

    metrics = {
        "dataset": "MetRec (official train/test split; 10% of train held out for validation)",
        "n_train": len(X_train_full),
        "n_validation_holdout": len(X_val),
        "n_test": len(X_test),
        "n_test_also_in_train": int(in_train.sum()),
        "chosen_C": best_c,
        "augmented_with_diacritic_free_copies": bool(use_aug),
        "validation": val_rows,
        "validation_accuracy_diacritics_removed": val_bare,
        "test": test_rows,
        "final_model_test": {
            "accuracy": accuracy(y_test, pred),
            "accuracy_95ci_bootstrap": [lo, hi],
            "macro_f1": macro_f1(y_test, pred),
            "accuracy_dedup_test": accuracy(y_test[keep], pred[keep]),
            "accuracy_if_diacritics_removed": accuracy(y_test, pred_bare),
            "accuracy_saved_float16_model": accuracy(y_test, pred_reloaded),
            "saved_vs_in_memory_agreement": agreement,
            "by_diacritization": by_diac,
        },
        "weights_size_mb": round(size_mb, 2),
        "reference_published_result": "AraMeter 5-layer BiGRU: 94.32% test accuracy (Al-shaibani et al., 2020) - not re-run here",
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    report = per_class_report(y_test, pred, len(METERS))
    with open(results / "per_meter.csv", "w", encoding="utf-8") as fh:
        fh.write("label,meter,support,precision,recall,f1\n")
        for r in report:
            fh.write(f"{r['label']},{METERS[r['label']]},{r['support']},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")
    np.savetxt(results / "confusion_matrix.csv", confusion(y_test, pred, len(METERS)), fmt="%d", delimiter=",",
               header=",".join(METERS), comments="")
    print(json.dumps(metrics["final_model_test"], indent=2))
    print(f"done in {metrics['runtime_seconds']} s, weights {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
