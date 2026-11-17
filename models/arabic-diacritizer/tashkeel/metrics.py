"""Diacritization metrics: DER and WER, with and without case endings.

* DER (diacritic error rate): share of Arabic letters whose predicted
  diacritic differs from the reference. Letters that carry no diacritic in the
  reference are counted too (the model must predict "none" for them).
* WER (word error rate): share of Arabic words with at least one wrong letter.
* "no case ending" variants ignore the last letter of every word, whose
  diacritic depends on grammar (i'rab) and is the hardest part.
"""
from __future__ import annotations

from .text import ARABIC_LETTERS


def _words(bare: str):
    i, n = 0, len(bare)
    while i < n:
        if bare[i] in ARABIC_LETTERS:
            j = i
            while j < n and bare[j] in ARABIC_LETTERS:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def evaluate(bare_lines, gold_labels, pred_labels) -> dict:
    """Compute DER/WER over lists of bare lines and per-character labels."""
    let = let_err = let_nce = let_nce_err = 0
    words = word_err = word_nce_err = 0
    for bare, gold, pred in zip(bare_lines, gold_labels, pred_labels):
        for s, e in _words(bare):
            words += 1
            wrong = wrong_nce = False
            for k in range(s, e):
                bad = gold[k] != pred[k]
                let += 1
                let_err += bad
                wrong |= bad
                if k != e - 1:
                    let_nce += 1
                    let_nce_err += bad
                    wrong_nce |= bad
            word_err += wrong
            word_nce_err += wrong_nce
    return {
        "letters": let,
        "words": words,
        "DER": let_err / max(let, 1),
        "DER_no_case_ending": let_nce_err / max(let_nce, 1),
        "WER": word_err / max(words, 1),
        "WER_no_case_ending": word_nce_err / max(words, 1),
    }
