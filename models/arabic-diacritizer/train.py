"""Train and evaluate the tashkeel diacritizer.

Usage:
    python train.py                 # full run (~12 min on 2 CPU cores)
    python train.py --quick         # small smoke run (a few minutes)

Steps:
 1. download the train / val / test splits (MIT-licensed benchmark)
 2. fit two baselines (letter majority, word lookup)
 3. train a linear classifier (SGD, hinge loss, averaged weights) for a few
    epochs, keeping the epoch with the best *validation* DER
 4. evaluate everything once on the held-out *test* split
 5. write weights to weights/ and reports/charts to results/
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier

from tashkeel.baselines import LetterMajority, WordLookup
from tashkeel.data import load_split
from tashkeel.metrics import evaluate
from tashkeel.model import Diacritizer, lines_to_xy
from tashkeel.text import ARABIC_LETTERS, label_name, split_text

HERE = Path(__file__).parent


def score(system, lines):
    bare, gold = zip(*(split_text(line) for line in lines))
    pred = system.predict_labels(list(bare))
    return evaluate(bare, gold, pred), pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n-bits", type=int, default=21, help="hash space = 2**n_bits columns")
    ap.add_argument("--alpha", type=float, default=1e-7, help="L2 regularisation strength")
    ap.add_argument("--chunk", type=int, default=5000, help="lines per SGD mini-batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    train = load_split("train", args.data_dir)
    val = load_split("val", args.data_dir)
    test = load_split("test", args.data_dir)
    if args.quick:
        train, val, test = train[:3000], val[:300], test[:300]
        args.epochs = min(args.epochs, 2)
        args.chunk = 1000
    print(f"lines: train={len(train)} val={len(val)} test={len(test)}")

    # ---- classes: every label seen at least 20 times in train
    counts = Counter()
    for line in train:
        counts.update(split_text(line)[1])
    classes = [c for c, n in counts.most_common() if n >= 20]
    cidx = {c: i for i, c in enumerate(classes)}
    print("classes:", [label_name(c) for c in classes])

    # ---- baselines
    baselines = {"letter_majority": LetterMajority().fit(train), "word_lookup": WordLookup().fit(train)}

    # ---- featurise training data once, in chunks
    order = rng.permutation(len(train))
    chunks = []
    for s in range(0, len(train), args.chunk):
        X, y = lines_to_xy([train[i] for i in order[s : s + args.chunk]], cidx, args.n_bits)
        chunks.append((X, y))
    n_rows = sum(X.shape[0] for X, _ in chunks)
    print(f"training letters: {n_rows:,}  ({time.time() - t0:.0f}s)")

    clf = SGDClassifier(loss="hinge", alpha=args.alpha, average=True, n_jobs=2, random_state=args.seed)
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        for k in rng.permutation(len(chunks)):
            X, y = chunks[k]
            clf.partial_fit(X, y, classes=np.arange(len(classes)))
        model = Diacritizer(classes, clf.coef_.astype(np.float32), clf.intercept_, args.n_bits)
        m, _ = score(model, val)
        history.append({"epoch": epoch, **{k: round(v, 5) for k, v in m.items() if k.startswith(("DER", "WER"))}})
        print(f"epoch {epoch}: val DER={m['DER']:.4f} WER={m['WER']:.4f}  ({time.time() - t0:.0f}s)")
        if best is None or m["DER"] < best[0]:
            best = (m["DER"], epoch, model)
    _, best_epoch, model = best
    del chunks

    # ---- save (int8-quantised) and reload, so we evaluate exactly what we release
    wpath = out / "weights" / "tashkeel-linear"
    model.save(wpath)
    released = Diacritizer.load(wpath)

    # ---- final, single evaluation on test
    report = {}
    systems = {**baselines, "tashkeel_linear (this model)": released}
    preds = {}
    for name, sys_ in systems.items():
        report[name] = {"val": score(sys_, val)[0]}
        report[name]["test"], preds[name] = score(sys_, test)
        t = report[name]["test"]
        print(f"TEST {name:30s} DER={t['DER']:.4f} DER-noCE={t['DER_no_case_ending']:.4f} "
              f"WER={t['WER']:.4f} WER-noCE={t['WER_no_case_ending']:.4f}")

    # per-class errors of the released model on test
    bare, gold = zip(*(split_text(line) for line in test))
    pred = preds["tashkeel_linear (this model)"]
    per_class = Counter(), Counter()
    for g_line, p_line, b_line in zip(gold, pred, bare):
        for ch, g, p in zip(b_line, g_line, p_line):
            if ch in ARABIC_LETTERS:
                per_class[0][g] += 1
                per_class[1][g] += g != p
    per_class_rows = [
        {"label": label_name(c), "count": per_class[0][c], "error_rate": round(per_class[1][c] / per_class[0][c], 4)}
        for c in classes if per_class[0][c]
    ]

    weight_files = sorted(p.name for p in (out / "weights").iterdir())
    size_mb = sum((out / "weights" / f).stat().st_size for f in weight_files) / 1e6
    meta = {
        "config": {k: v for k, v in vars(args).items() if k not in ("data_dir", "out")},
        "best_epoch_on_val": best_epoch,
        "val_history": history,
        "classes": [label_name(c) for c in classes],
        "n_train_lines": len(train),
        "n_train_letters": n_rows,
        "weights_size_mb": round(size_mb, 2),
        "runtime_seconds": round(time.time() - t0),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "results": report,
        "per_class_test": per_class_rows,
    }
    (out / "results" / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    write_markdown(meta, out / "results" / "metrics.md")
    write_samples(test[:8], released, out / "results" / "samples.md")
    try:
        make_chart(report, out / "results" / "der_wer_comparison.png")
    except ImportError:
        print("matplotlib not installed; skipping chart")
    print(f"done in {time.time() - t0:.0f}s; weights {size_mb:.1f} MB")


def write_markdown(meta: dict, path: Path) -> None:
    lines = [
        "# Test-set results (2,500 held-out lines)",
        "",
        "Lower is better. DER = share of letters with a wrong diacritic; WER = share of words with at least one wrong letter.",
        "'no CE' = ignoring the last letter of each word (the grammatical case ending).",
        "",
        "| System | DER | DER (no CE) | WER | WER (no CE) |",
        "|---|---|---|---|---|",
    ]
    for name, r in meta["results"].items():
        t = r["test"]
        lines.append(f"| {name} | {t['DER']:.2%} | {t['DER_no_case_ending']:.2%} | {t['WER']:.2%} | {t['WER_no_case_ending']:.2%} |")
    lines += ["", f"Best epoch on validation: {meta['best_epoch_on_val']}; weights {meta['weights_size_mb']} MB; "
              f"run time {meta['runtime_seconds']} s.", "", "## Error rate per diacritic (this model, test)", "",
              "| Gold diacritic | Letters | Error rate |", "|---|---|---|"]
    for row in meta["per_class_test"]:
        lines.append(f"| {row['label']} | {row['count']:,} | {row['error_rate']:.2%} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_samples(lines: list[str], model: Diacritizer, path: Path) -> None:
    from tashkeel.text import strip_diacritics

    rows = ["# Sample predictions (first test lines, truncated)", ""]
    for line in lines:
        line = " ".join(line.split()[:14])
        bare = strip_diacritics(line)
        rows += [f"- input:      {bare}", f"  - model:     {model.diacritize(bare)}", f"  - reference: {line}", ""]
    path.write_text("\n".join(rows), encoding="utf-8")


def make_chart(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report)
    metrics = [("DER", "DER"), ("DER_no_case_ending", "DER (no case ending)"), ("WER", "WER"),
               ("WER_no_case_ending", "WER (no case ending)")]
    colors = ["#9aa5b1", "#5b8def", "#1b9e77"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    width = 0.8 / len(names)
    x = np.arange(len(metrics))
    for i, name in enumerate(names):
        vals = [report[name]["test"][k] * 100 for k, _ in metrics]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, label=name, color=colors[i % 3])
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x, [m for _, m in metrics])
    ax.set_ylabel("error rate on test set (%)  - lower is better")
    ax.set_title("Arabic diacritization: model vs baselines (2,500 test lines)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
