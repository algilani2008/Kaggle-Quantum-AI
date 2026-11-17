"""Vectorised feature extraction for per-letter diacritic prediction.

Every Arabic letter becomes one row of a sparse binary matrix. Features are
hashed (the "hashing trick") into ``2**n_bits`` columns, so the model has a
fixed size no matter how many distinct words or character n-grams exist.

Features for a letter at position p:
  * single characters in a window of +-5 around p
  * character 2-, 3-, 4- and 5-grams around p
  * the whole (bare) word together with the letter's position inside it,
    which lets the model memorise frequent words
  * the previous / next word together with the current word and whether p is
    the last letter (case endings depend on the neighbouring words)
  * letter identity x position in word x word length
"""
from __future__ import annotations

import zlib

import numpy as np
import scipy.sparse as sp

from .text import ARABIC_LETTERS

WINDOW = 5
PAD = 0
_LETTERS = sorted(ARABIC_LETTERS)
_LETTER_ID = {ch: i + 1 for i, ch in enumerate(_LETTERS)}
SPACE_ID = len(_LETTERS) + 1
DIGIT_ID = SPACE_ID + 1
PUNCT_ID = SPACE_ID + 2
OTHER_ID = SPACE_ID + 3

_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xBF58476D1CE4E5B9)
_M3 = np.uint64(0x94D049BB133111EB)


def _char_id(ch: str) -> int:
    if ch in _LETTER_ID:
        return _LETTER_ID[ch]
    if ch == "\n":
        return PAD
    if ch.isspace():
        return SPACE_ID
    if ch.isdigit():
        return DIGIT_ID
    if not ch.isalnum():
        return PUNCT_ID
    return OTHER_ID


def _mix(salt: int, *cols: np.ndarray) -> np.ndarray:
    """Deterministic 64-bit hash of several integer columns (splitmix-style)."""
    with np.errstate(over="ignore"):
        x = np.full(cols[0].shape, np.uint64(salt) * _M1 + np.uint64(1), dtype=np.uint64)
        for c in cols:
            x = (x ^ c.astype(np.uint64)) * _M2
            x ^= x >> np.uint64(31)
        x = x * _M3
        x ^= x >> np.uint64(29)
    return x


def _word_hashes(text: str, is_letter: np.ndarray):
    """Per-character hash of the containing word, plus neighbouring words."""
    n = len(text)
    starts = np.flatnonzero(is_letter & ~np.r_[False, is_letter[:-1]])
    ends = np.flatnonzero(is_letter & ~np.r_[is_letter[1:], False]) + 1
    whash = np.array([zlib.crc32(text[s:e].encode("utf-8")) + 1 for s, e in zip(starts, ends)], dtype=np.uint64)
    # neighbouring words, only within the same line
    newline_cum = np.cumsum(np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32) == 10)
    line_of_word = newline_cum[starts] if len(starts) else np.array([], dtype=np.int64)
    prev_h = np.r_[np.uint64(0), whash[:-1]] if len(whash) else whash
    next_h = np.r_[whash[1:], np.uint64(0)] if len(whash) else whash
    if len(whash):
        prev_ok = np.r_[False, line_of_word[1:] == line_of_word[:-1]]
        next_ok = np.r_[line_of_word[1:] == line_of_word[:-1], False]
        prev_h = np.where(prev_ok, prev_h, np.uint64(0))
        next_h = np.where(next_ok, next_h, np.uint64(0))
    seg = np.full(n, -1, dtype=np.int64)
    lengths = ends - starts
    seg_idx = np.repeat(np.arange(len(starts)), lengths)
    pos_idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) if len(starts) else np.array([], dtype=np.int64)
    seg[pos_idx] = seg_idx
    pos_in_word = np.zeros(n, dtype=np.int64)
    pos_from_end = np.zeros(n, dtype=np.int64)
    wlen = np.zeros(n, dtype=np.int64)
    pos_in_word[pos_idx] = pos_idx - starts[seg_idx]
    pos_from_end[pos_idx] = ends[seg_idx] - 1 - pos_idx
    wlen[pos_idx] = lengths[seg_idx]
    return seg, whash, prev_h, next_h, pos_in_word, pos_from_end, wlen


def featurize(lines: list[str], n_bits: int = 20):
    """Build the sparse feature matrix for every Arabic letter in ``lines``.

    ``lines`` must be bare (undiacritized) text without newlines.
    Returns ``(X, line_index, char_index)`` where row i of X describes
    character ``char_index[i]`` of line ``line_index[i]``.
    """
    text = "\n".join(lines) + "\n"
    codes = np.array([_char_id(ch) for ch in text], dtype=np.int64)
    is_letter = np.array([ch in ARABIC_LETTERS for ch in text], dtype=bool)
    n = len(text)
    padded = np.r_[np.zeros(WINDOW, dtype=np.int64), codes, np.zeros(WINDOW, dtype=np.int64)]
    seg, whash, prev_h, next_h, pin, pend, wlen = _word_hashes(text, is_letter)

    p = np.flatnonzero(is_letter)
    c = {o: padded[p + WINDOW + o] for o in range(-WINDOW, WINDOW + 1)}
    s = seg[p]
    wh, ph, nh = whash[s], prev_h[s], next_h[s]
    pi, pe, wl = np.minimum(pin[p], 6), np.minimum(pend[p], 6), np.minimum(wlen[p], 10)
    last = (pend[p] == 0).astype(np.int64)

    cols = [np.zeros(len(p), dtype=np.uint64)]  # bias
    salt = 1
    for o in range(-WINDOW, WINDOW + 1):  # unigrams
        cols.append(_mix(salt, c[o])); salt += 1
    for o in range(-3, 3):  # bigrams
        cols.append(_mix(salt, c[o], c[o + 1])); salt += 1
    for o in range(-2, 1):  # trigrams
        cols.append(_mix(salt, c[o], c[o + 1], c[o + 2])); salt += 1
    for o in range(-3, 1):  # 4-grams
        cols.append(_mix(salt, c[o], c[o + 1], c[o + 2], c[o + 3])); salt += 1
    cols.append(_mix(salt, c[-2], c[-1], c[0], c[1], c[2])); salt += 1  # 5-gram
    cols.append(_mix(salt, wh, pin[p])); salt += 1  # word x position
    cols.append(_mix(salt, ph, wh, pin[p])); salt += 1  # previous word x word x position
    cols.append(_mix(salt, ph, c[0], pe, last)); salt += 1
    cols.append(_mix(salt, nh, c[0], pe, last)); salt += 1
    cols.append(_mix(salt, wh, nh, last)); salt += 1
    cols.append(_mix(salt, c[0], pi, pe, wl)); salt += 1

    mask = np.uint64((1 << n_bits) - 1)
    idx = np.stack([col & mask for col in cols], axis=1).astype(np.int32)
    idx[:, 0] = 0  # bias always column 0
    n_feat = idx.shape[1]
    X = sp.csr_matrix(
        (np.ones(idx.size, dtype=np.float32), idx.ravel(), np.arange(0, idx.size + 1, n_feat)),
        shape=(len(p), 1 << n_bits),
    )
    X.sum_duplicates()

    # map flat positions back to (line, char)
    line_starts = np.r_[0, np.cumsum([len(line) + 1 for line in lines])[:-1]] if lines else np.array([0])
    line_index = np.searchsorted(line_starts, p, side="right") - 1
    char_index = p - line_starts[line_index]
    return X, line_index, char_index
