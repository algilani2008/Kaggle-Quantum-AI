"""The diacritizer model: a linear classifier over hashed character features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import featurize
from .text import FATHATAN, DAMMATAN, KASRATAN, join_text, split_text, strip_diacritics

_TANWEEN = (FATHATAN, DAMMATAN, KASRATAN)


class Diacritizer:
    """Predicts one diacritic label per Arabic letter.

    The weights are a (n_classes x 2**n_bits) matrix stored as int8 with one
    float scale per class, which keeps the released file small.
    """

    def __init__(self, classes: list[str], weights: np.ndarray, bias: np.ndarray, n_bits: int):
        self.classes = list(classes)
        self.weights = weights  # float32 (n_classes, 2**n_bits)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.n_bits = n_bits
        # classes containing tanween (nunation)
        self._tanween_mask = np.array([any(t in c for t in _TANWEEN) for c in self.classes])

    # ------------------------------------------------------------------ predict
    def scores(self, X) -> np.ndarray:
        return np.asarray(X @ self.weights.T) + self.bias

    def predict_labels(self, bare_lines: list[str]) -> list[list[str]]:
        """Return one label per character for each bare line."""
        labels = [[""] * len(line) for line in bare_lines]
        if not bare_lines:
            return labels
        X, li, ci = featurize(bare_lines, self.n_bits)
        if X.shape[0] == 0:
            return labels
        sc = self.scores(X)
        # tanween (nunation) only on the last or second-to-last letter of a word
        # (it sits before a final alef / alef maqsura, e.g. رَجُلًا, هُدًى)
        not_last = np.array([_too_far_from_end(bare_lines[a], b) for a, b in zip(li, ci)])
        sc[np.ix_(not_last, self._tanween_mask)] = -np.inf
        pred = sc.argmax(axis=1)
        for a, b, k in zip(li, ci, pred):
            labels[a][b] = self.classes[k]
        return labels

    def diacritize(self, text: str | list[str]):
        """Add diacritics to text (existing diacritics are removed first)."""
        single = isinstance(text, str)
        lines = [text] if single else list(text)
        out = []
        for chunk in lines:
            parts = strip_diacritics(chunk).split("\n")
            labs = self.predict_labels(parts)
            out.append("\n".join(join_text(p, lab) for p, lab in zip(parts, labs)))
        return out[0] if single else out

    # ---------------------------------------------------------------- save/load
    def save(self, path: str | Path) -> None:
        path = Path(path)
        scale = np.abs(self.weights).max(axis=1) / 127.0
        scale[scale == 0] = 1.0
        q = np.round(self.weights / scale[:, None]).astype(np.int8)
        np.savez_compressed(path, q=q, scale=scale.astype(np.float32), bias=self.bias)
        meta = {"classes": self.classes, "n_bits": self.n_bits, "format": "tashkeel-linear-v1"}
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Diacritizer":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        z = np.load(path.with_suffix(".npz"))
        w = z["q"].astype(np.float32) * z["scale"][:, None]
        return cls(meta["classes"], w, z["bias"], meta["n_bits"])


def _too_far_from_end(line: str, i: int) -> bool:
    """True if letter i is neither the last nor the second-to-last letter of its word."""
    from .text import ARABIC_LETTERS

    def is_letter(k: int) -> bool:
        return k < len(line) and line[k] in ARABIC_LETTERS

    return is_letter(i + 1) and is_letter(i + 2)


def lines_to_xy(diacritized_lines: list[str], class_index: dict[str, int], n_bits: int):
    """Turn diacritized lines into (X, y) for training."""
    bare, labs = zip(*(split_text(line) for line in diacritized_lines)) if diacritized_lines else ((), ())
    X, li, ci = featurize(list(bare), n_bits)
    y = np.array([class_index.get(labs[a][b], class_index[""]) for a, b in zip(li, ci)], dtype=np.int64)
    return X, y
