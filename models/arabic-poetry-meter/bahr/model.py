"""MeterClassifier: character n-grams + rhythm-skeleton n-grams + a linear SVM.

The model is saved as plain JSON (vocabularies, settings) and a compressed
NumPy archive (idf weights, linear weights). No pickle is used, so loading a
downloaded model never executes code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from .features import letter_counts, marked_skeleton
from .labels import METERS, METERS_AR

FORMAT_VERSION = 1
LENGTH_CAP = 40
DEFAULT_BLOCKS = ("text", "rhythm", "length")

# Settings for each TF-IDF block: (preprocessor name, n-gram range, max features, min_df)
BLOCK_SETTINGS = {
    "text": {"ngram_range": (1, 5), "max_features": 100_000, "min_df": 2},
    "rhythm": {"ngram_range": (2, 8), "max_features": 150_000, "min_df": 3},
}


def _prep(block: str, verses: list[str]) -> list[str]:
    if block == "text":
        return [v.replace("#", " # ") for v in verses]
    if block == "rhythm":
        return [marked_skeleton(v) for v in verses]
    raise ValueError(block)


def _length_matrix(verses: list[str]) -> sp.csr_matrix:
    """One-hot of letter counts per hemistich: 2 blocks of LENGTH_CAP+1 columns."""
    rows, cols = [], []
    for i, v in enumerate(verses):
        for half, count in enumerate(letter_counts(v, LENGTH_CAP)):
            rows.append(i)
            cols.append(half * (LENGTH_CAP + 1) + count)
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(verses), 2 * (LENGTH_CAP + 1)))


class MeterClassifier:
    """Predict one of 14 classical Arabic meters for a verse ``"hemistich 1 # hemistich 2"``."""

    def __init__(self, blocks=DEFAULT_BLOCKS, C: float = 0.5, block_settings=None):
        unknown = set(blocks) - {"text", "rhythm", "length"}
        if unknown or not blocks:
            raise ValueError(f"bad blocks: {blocks}")
        self.blocks = tuple(blocks)
        self.C = C
        self.block_settings = {k: dict(v) for k, v in (block_settings or BLOCK_SETTINGS).items()}
        self.vocab_: dict[str, dict[str, int]] = {}
        self.idf_: dict[str, np.ndarray] = {}
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    # ---------- features ----------
    def _tfidf(self, block: str, verses: list[str]) -> sp.csr_matrix:
        s = self.block_settings[block]
        counter = CountVectorizer(
            analyzer="char",
            ngram_range=tuple(s["ngram_range"]),
            lowercase=False,
            vocabulary=self.vocab_[block],
        )
        X = counter.transform(_prep(block, verses)).astype(np.float64)
        X.data = 1.0 + np.log(X.data)  # sublinear tf
        X = X @ sp.diags(self.idf_[block])
        return normalize(X, norm="l2").tocsr()

    def transform(self, verses: list[str]) -> sp.csr_matrix:
        parts = []
        for block in self.blocks:
            parts.append(_length_matrix(verses) if block == "length" else self._tfidf(block, verses))
        return sp.hstack(parts).tocsr()

    # ---------- training ----------
    def fit(self, verses: list[str], y) -> "MeterClassifier":
        y = np.asarray(y)
        for block in self.blocks:
            if block == "length":
                continue
            s = self.block_settings[block]
            vec = TfidfVectorizer(
                analyzer="char",
                ngram_range=tuple(s["ngram_range"]),
                max_features=s["max_features"],
                min_df=s["min_df"],
                sublinear_tf=True,
                lowercase=False,
            )
            vec.fit(_prep(block, verses))
            self.vocab_[block] = {k: int(v) for k, v in vec.vocabulary_.items()}
            self.idf_[block] = vec.idf_.astype(np.float64)
        X = self.transform(verses)
        svm = LinearSVC(C=self.C, random_state=0).fit(X, y)
        self.coef_ = svm.coef_.astype(np.float32)
        self.intercept_ = svm.intercept_.astype(np.float32)
        self.classes_ = svm.classes_
        return self

    # ---------- prediction ----------
    def decision_function(self, verses: list[str]) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not trained")
        X = self.transform(list(verses))
        scores = np.asarray(X @ self.coef_.astype(np.float64).T) + self.intercept_
        if scores.shape[1] == 1:  # binary LinearSVC stores one row
            scores = np.hstack([-scores, scores])
        return scores

    def predict(self, verses: list[str]) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(verses), axis=1)]

    def top_k(self, verse: str, k: int = 3) -> list[dict]:
        """Best ``k`` meters for one verse with their raw SVM scores (not probabilities)."""
        scores = self.decision_function([verse])[0]
        order = np.argsort(-scores)[:k]
        out = []
        for i in order:
            label = int(self.classes_[i])
            name = METERS[label] if 0 <= label < len(METERS) else str(label)
            out.append({"label": label, "meter": name, "meter_ar": METERS_AR.get(name, name), "score": float(scores[i])})
        return out

    # ---------- persistence (no pickle) ----------
    def save(self, directory: str | Path, half_precision: bool = True) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        config = {
            "format_version": FORMAT_VERSION,
            "blocks": list(self.blocks),
            "C": self.C,
            "block_settings": {k: {**v, "ngram_range": list(v["ngram_range"])} for k, v in self.block_settings.items()},
            "classes": [int(c) for c in self.classes_],
            "meters": METERS,
            "half_precision": half_precision,
        }
        (d / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        for block, vocab in self.vocab_.items():
            terms = [None] * len(vocab)
            for term, idx in vocab.items():
                terms[idx] = term
            (d / f"vocab_{block}.json").write_text(json.dumps(terms, ensure_ascii=False), encoding="utf-8")
        arrays = {f"idf_{b}": v.astype(np.float32) for b, v in self.idf_.items()}
        arrays["coef"] = self.coef_.astype(np.float16 if half_precision else np.float32)
        arrays["intercept"] = self.intercept_.astype(np.float32)
        np.savez_compressed(d / "weights.npz", **arrays)

    @classmethod
    def load(cls, directory: str | Path) -> "MeterClassifier":
        d = Path(directory)
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if config.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported model format")
        settings = {k: {**v, "ngram_range": tuple(v["ngram_range"])} for k, v in config["block_settings"].items()}
        model = cls(blocks=config["blocks"], C=config["C"], block_settings=settings)
        with np.load(d / "weights.npz", allow_pickle=False) as w:
            for block in model.blocks:
                if block == "length":
                    continue
                terms = json.loads((d / f"vocab_{block}.json").read_text(encoding="utf-8"))
                model.vocab_[block] = {t: i for i, t in enumerate(terms)}
                model.idf_[block] = w[f"idf_{block}"].astype(np.float64)
            model.coef_ = w["coef"].astype(np.float32)
            model.intercept_ = w["intercept"].astype(np.float32)
        model.classes_ = np.array(config["classes"])
        return model
