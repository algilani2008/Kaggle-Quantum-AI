"""Command-line tool: add diacritics (tashkeel) to Arabic text.

Examples:
    python diacritize.py "ذهب الطالب الى المدرسة"
    echo "قال الشيخ" | python diacritize.py
    python diacritize.py -i input.txt -o output.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tashkeel import Diacritizer

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "tashkeel-linear"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Add Arabic diacritics to text.")
    ap.add_argument("text", nargs="?", help="text to diacritize (otherwise read --input or stdin)")
    ap.add_argument("-i", "--input", help="UTF-8 text file to read")
    ap.add_argument("-o", "--output", help="file to write (default: stdout)")
    ap.add_argument("-w", "--weights", default=str(DEFAULT_WEIGHTS), help="weights path without extension")
    args = ap.parse_args(argv)

    if args.text is not None:
        text = args.text
    elif args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    model = Diacritizer.load(args.weights)
    result = model.diacritize(text.rstrip("\n"))
    if args.output:
        Path(args.output).write_text(result + "\n", encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
