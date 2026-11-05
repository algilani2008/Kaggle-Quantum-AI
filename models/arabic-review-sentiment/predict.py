#!/usr/bin/env python3
"""Classify Arabic reviews with the model saved by train.py.

Example:  python predict.py "الكتاب رائع وأسلوبه ممتع"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

MODEL = Path(__file__).resolve().parent / "artifacts" / "model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify Arabic reviews as positive or negative.")
    parser.add_argument("texts", nargs="+", help="review text(s), quoted")
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f"{args.model} not found; run train.py first")
    # Only load a model file you trained yourself: joblib can run code on load.
    model = joblib.load(args.model)
    for text, score in zip(args.texts, model.predict_proba(args.texts)):
        label = "positive" if score >= model.threshold else "negative"
        print(f"{label}\tscore={score:.2f}\t{text}")


if __name__ == "__main__":
    main()
