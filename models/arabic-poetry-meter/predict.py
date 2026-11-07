"""Predict the meter of one or more Arabic verses with the saved model.

Examples
    python predict.py "قفا نبك من ذكرى حبيب ومنزل # بسقط اللوى بين الدخول فحومل"
    python predict.py --file my_verses.txt      (one verse per line, halves separated by #)

The scores are raw linear-SVM margins, not probabilities: compare them with
each other, not with a fixed threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bahr.model import MeterClassifier

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verse", nargs="*", help='verse text, e.g. "hemistich one # hemistich two"')
    ap.add_argument("--file", help="text file with one verse per line")
    ap.add_argument("--weights", default=str(HERE / "weights"))
    ap.add_argument("-k", type=int, default=3, help="how many candidate meters to show")
    args = ap.parse_args(argv)

    verses = list(args.verse)
    if args.file:
        verses += [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not verses:
        ap.print_help()
        return 1
    model = MeterClassifier.load(args.weights)
    for v in verses:
        if "#" not in v:
            print("note: no '#' between the two hemistichs; accuracy is lower on half verses", file=sys.stderr)
        print(v)
        for rank, c in enumerate(model.top_k(v, args.k), 1):
            print(f"  {rank}. {c['meter_ar']} ({c['meter']})  score={c['score']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
