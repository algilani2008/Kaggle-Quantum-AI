"""Feature sets and classifiers compared in train.py."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .textnorm import normalize_arabic

_TOKEN = r"(?u)\b\w+\b"  # keeps one-letter words


def word_tfidf(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word", token_pattern=_TOKEN, ngram_range=(1, 2), min_df=3,
        max_features=max_features, sublinear_tf=True,
    )


def char_tfidf(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3,
        max_features=max_features, sublinear_tf=True,
    )


class TextFeatures:
    """Optional normalization plus one or more fitted vectorizers.

    A plain class instead of a sklearn Pipeline, so that one feature matrix can
    be reused across the regularization grid, and so a saved model only needs
    this package to load.
    """

    def __init__(self, vectorizers: list, normalize: bool = True):
        self.vectorizers = vectorizers
        self.normalize = normalize

    def _clean(self, texts):
        return [normalize_arabic(t) for t in texts] if self.normalize else list(texts)

    def fit_transform(self, texts):
        cleaned = self._clean(texts)
        return hstack([v.fit_transform(cleaned) for v in self.vectorizers]).tocsr()

    def transform(self, texts):
        cleaned = self._clean(texts)
        return hstack([v.transform(cleaned) for v in self.vectorizers]).tocsr()

    def feature_names(self) -> list[str]:
        return [name for v in self.vectorizers for name in v.get_feature_names_out()]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kinds: tuple[str, ...]
    normalize: bool = True

    def build(self, max_features: int = 400_000) -> TextFeatures:
        makers = {"word": word_tfidf, "char": char_tfidf}
        return TextFeatures([makers[k](max_features) for k in self.kinds], self.normalize)


FEATURE_SPECS = [
    FeatureSpec("word_raw", ("word",), normalize=False),
    FeatureSpec("word", ("word",)),
    FeatureSpec("char", ("char",)),
    FeatureSpec("word_char", ("word", "char")),
]


def naive_bayes_features() -> TextFeatures:
    """Plain single-word counts: the classic text baseline."""
    return TextFeatures([CountVectorizer(token_pattern=_TOKEN, min_df=2)])


class PlattCalibrator:
    """Maps raw model probabilities to honest ones.

    Balanced class weights push scores towards the rare class. A one-feature
    logistic regression on the log-odds, fitted on validation data, corrects
    that without changing the ranking of reviews.
    """

    def __init__(self):
        self.model = LogisticRegression(C=1e6)

    @staticmethod
    def _logit(prob) -> np.ndarray:
        clipped = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, prob, y):
        self.model.fit(self._logit(prob), y)
        return self

    def transform(self, prob) -> np.ndarray:
        return self.model.predict_proba(self._logit(prob))[:, 1]


class SentimentModel:
    """Fitted features and classifier, plus calibration and decision threshold."""

    def __init__(self, features: TextFeatures, classifier, threshold: float = 0.5, calibrator=None):
        self.features = features
        self.classifier = classifier
        self.threshold = threshold
        self.calibrator = calibrator

    def raw_scores(self, texts) -> np.ndarray:
        return self.classifier.predict_proba(self.features.transform(texts))[:, 1]

    def predict_proba(self, texts) -> np.ndarray:
        scores = self.raw_scores(texts)
        return self.calibrator.transform(scores) if self.calibrator else scores

    def predict(self, texts) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(int)
