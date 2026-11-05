"""Light Arabic text normalization.

The goal is to fold spelling variants of the same word together without
removing words. Negation words such as "لا" and "ما" are kept on purpose, so
this module has no stop-word list.
"""

from __future__ import annotations

import re

_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ALEF = re.compile(r"[آأإٱ]")  # آ أ إ ٱ -> ا
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_REPEAT = re.compile(r"([^\d\s])\1{2,}")  # digits are left alone: 2000 stays 2000
_SPACE = re.compile(r"\s+")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_arabic(text: str) -> str:
    """Return ``text`` with common Arabic spelling variation folded.

    Steps: drop URLs, diacritics and the elongation mark; unify alef forms,
    alef maqsura (ى -> ي) and ta marbuta (ة -> ه); convert Arabic-Indic digits;
    shorten any non-digit character repeated three or more times to two ("جمييييل" ->
    "جمييل"); lowercase Latin letters; collapse whitespace.
    """
    text = _URL.sub(" ", text)
    text = _TASHKEEL.sub("", text).replace(_TATWEEL, "")
    text = _ALEF.sub("ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.translate(_DIGITS)
    text = _REPEAT.sub(r"\1\1", text)
    return _SPACE.sub(" ", text).strip().lower()
