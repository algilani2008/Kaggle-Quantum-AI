"""Arabic text helpers: split diacritized text into (letters, labels) and back.

A *label* is the diacritic sequence written on one base letter, in a
canonical order (shadda first, then at most one vowel / tanween / sukun).
The empty string means "no diacritic".
"""
from __future__ import annotations

FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
FATHATAN, DAMMATAN, KASRATAN = "ً", "ٌ", "ٍ"
SUKUN, SHADDA = "ْ", "ّ"

DIACRITICS = frozenset([FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN, SUKUN, SHADDA])
# Base letters that can carry a diacritic (hamza forms, alef forms, 28 letters, ta marbuta, alef maqsura).
ARABIC_LETTERS = frozenset(
    [chr(c) for c in range(0x0621, 0x063B)] + [chr(c) for c in range(0x0641, 0x064B)] + ["ٱ"]
)

_ORDER = {SHADDA: 0, FATHA: 1, DAMMA: 1, KASRA: 1, FATHATAN: 1, DAMMATAN: 1, KASRATAN: 1, SUKUN: 1}

# Human-readable names, used in reports.
LABEL_NAMES = {
    "": "none",
    FATHA: "fatha",
    DAMMA: "damma",
    KASRA: "kasra",
    FATHATAN: "fathatan",
    DAMMATAN: "dammatan",
    KASRATAN: "kasratan",
    SUKUN: "sukun",
    SHADDA: "shadda",
}


def label_name(label: str) -> str:
    """Readable name for a label, e.g. shadda+fatha."""
    return "+".join(LABEL_NAMES.get(ch, hex(ord(ch))) for ch in label) or "none"


def canonical(marks: str) -> str:
    """Return diacritics in canonical order with duplicates removed."""
    seen = []
    for ch in sorted(marks, key=lambda c: _ORDER.get(c, 2)):
        if ch not in seen:
            seen.append(ch)
    return "".join(seen)


def strip_diacritics(text: str) -> str:
    """Remove all harakat/tanween/shadda/sukun from text."""
    return "".join(ch for ch in text if ch not in DIACRITICS)


def split_text(text: str) -> tuple[str, list[str]]:
    """Split diacritized text into bare text and one label per bare character.

    Non-letters always get the empty label. Diacritics that do not follow an
    Arabic letter are dropped.
    """
    bare: list[str] = []
    labels: list[str] = []
    for ch in text:
        if ch in DIACRITICS:
            if bare and bare[-1] in ARABIC_LETTERS:
                labels[-1] += ch
            continue
        bare.append(ch)
        labels.append("")
    return "".join(bare), [canonical(lab) for lab in labels]


def join_text(bare: str, labels: list[str]) -> str:
    """Inverse of :func:`split_text`."""
    if len(bare) != len(labels):
        raise ValueError("bare text and labels must have the same length")
    return "".join(ch + (lab if ch in ARABIC_LETTERS else "") for ch, lab in zip(bare, labels))
