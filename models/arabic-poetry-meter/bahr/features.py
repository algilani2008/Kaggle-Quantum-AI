"""Text features for meter classification.

Arabic meter is a pattern of short and long syllables. A full prosodic (Arudi)
transcription needs complete diacritics and many rules, so this module uses a
cheap approximation, the *rhythm skeleton*: every character is mapped to a
small symbol set that keeps what matters for rhythm and drops what does not
(which letter it is).

    C  consonant letter            L  long-vowel / madd letter (ا و ي ى آ)
    v  short vowel mark (fatha, damma, kasra)
    n  tanween                     s  sukun          d  shadda

Each hemistich (شطر) is wrapped in ``^`` ... ``$`` so that n-grams can see the
start and, most importantly, the ending of each half, which is very
characteristic of a meter. Spaces become ``_``.
"""

from __future__ import annotations

import re

SEPARATOR = "#"
LONG_LETTERS = set("اوىيآ")
SHORT_VOWELS = set("َُِ")  # fatha, damma, kasra
TANWEEN = set("ًٌٍ")
SUKUN = "ْ"
SHADDA = "ّ"
_DIACRITICS = re.compile("[ً-ْٰـ]")  # harakat, dagger alef, tatweel
_ARABIC_LETTER = re.compile("[ء-ي]")


def strip_diacritics(text: str) -> str:
    """Remove harakat, shadda, sukun, tanween, dagger alef and tatweel."""
    return _DIACRITICS.sub("", text)


def halves(verse: str) -> list[str]:
    """Split a verse (بيت) into its two hemistichs on ``#``.

    A line without exactly one separator is treated as a single hemistich and
    an empty second half.
    """
    parts = verse.split(SEPARATOR)
    if len(parts) == 2:
        return [parts[0].strip(), parts[1].strip()]
    return [verse.replace(SEPARATOR, " ").strip(), ""]


def rhythm_skeleton(text: str) -> str:
    """Map one hemistich to the rhythm alphabet described in the module docstring."""
    out = []
    for ch in text:
        if ch == " ":
            if out and out[-1] != "_":
                out.append("_")
        elif ch in SHORT_VOWELS:
            out.append("v")
        elif ch in TANWEEN:
            out.append("n")
        elif ch == SUKUN:
            out.append("s")
        elif ch == SHADDA:
            out.append("d")
        elif ch in LONG_LETTERS:
            out.append("L")
        elif _ARABIC_LETTER.match(ch):
            out.append("C")
    return "".join(out).strip("_")


def marked_skeleton(verse: str) -> str:
    """Skeleton of both hemistichs, each wrapped in ``^...$``."""
    return " ".join("^" + rhythm_skeleton(h) + "$" for h in halves(verse))


def letter_counts(verse: str, cap: int = 40) -> list[int]:
    """Number of Arabic letters in each hemistich (capped), a crude length signal."""
    return [min(len(_ARABIC_LETTER.findall(h)), cap) for h in halves(verse)]


def diacritic_density(verse: str) -> float:
    """Share of Arabic letters followed by a diacritic mark (0 = bare text)."""
    letters = len(_ARABIC_LETTER.findall(verse))
    if letters == 0:
        return 0.0
    marks = len(re.findall("[ً-ْ]", verse))
    return min(marks / letters, 1.0)
