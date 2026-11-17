"""Simple reference systems the trained model must beat."""
from __future__ import annotations

from collections import Counter, defaultdict

from .metrics import _words
from .text import ARABIC_LETTERS, split_text


class LetterMajority:
    """Give every letter the diacritic it most often has in training."""

    def fit(self, diacritized_lines):
        counts = defaultdict(Counter)
        for line in diacritized_lines:
            bare, labs = split_text(line)
            for ch, lab in zip(bare, labs):
                if ch in ARABIC_LETTERS:
                    counts[ch][lab] += 1
        self.best = {ch: c.most_common(1)[0][0] for ch, c in counts.items()}
        return self

    def predict_labels(self, bare_lines):
        return [[self.best.get(ch, "") for ch in line] for line in bare_lines]


class WordLookup:
    """Memorise the most frequent diacritized form of every training word.

    Unknown words fall back to :class:`LetterMajority`. This is a strong,
    common baseline: most words in a corpus have been seen before.
    """

    def fit(self, diacritized_lines):
        forms = defaultdict(Counter)
        for line in diacritized_lines:
            bare, labs = split_text(line)
            for s, e in _words(bare):
                forms[bare[s:e]][tuple(labs[s:e])] += 1
        self.lookup = {w: c.most_common(1)[0][0] for w, c in forms.items()}
        self.fallback = LetterMajority().fit(diacritized_lines)
        return self

    def predict_labels(self, bare_lines):
        out = self.fallback.predict_labels(bare_lines)
        for bare, labs in zip(bare_lines, out):
            for s, e in _words(bare):
                form = self.lookup.get(bare[s:e])
                if form is not None:
                    labs[s:e] = list(form)
        return out
