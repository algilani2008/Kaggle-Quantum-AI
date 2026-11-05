import numpy as np
import pandas as pd

from arsent.models import FeatureSpec, PlattCalibrator, SentimentModel
from train import by_length, fit_model, fit_naive_bayes, select_model, top_features

GOOD = ["كتاب رائع جدا", "قصة جميلة وممتعة", "اسلوب ممتاز ورائع", "احببت هذا الكتاب الجميل"]
BAD = ["كتاب سيء جدا", "قصة مملة وسيئة", "اسلوب ضعيف وممل", "كرهت هذا الكتاب السيء"]


def tiny_frame(copies=6):
    texts = (GOOD + BAD) * copies
    labels = ([1] * 4 + [0] * 4) * copies
    return pd.DataFrame({"text": texts, "label": labels, "book_id": range(len(texts))})


def test_fit_model_learns_a_simple_rule():
    model = fit_model(FeatureSpec("word", ("word",)), 10.0, tiny_frame())
    assert isinstance(model, SentimentModel)
    assert model.predict(["كتاب رائع", "كتاب سيء"]).tolist() == [1, 0]


def test_naive_bayes_baseline_runs():
    model = fit_naive_bayes(tiny_frame())
    assert model.predict(["قصة جميلة", "قصة مملة"]).tolist() == [1, 0]


def test_select_model_returns_best_first():
    frame = tiny_frame()
    grid = select_model(frame, frame, specs=[FeatureSpec("word", ("word",))], grid=(1.0, 10.0))
    assert len(grid) == 2
    assert grid["val_macro_f1"].is_monotonic_decreasing


def test_top_features_are_signed_correctly():
    model = fit_model(FeatureSpec("word", ("word",)), 10.0, tiny_frame())
    top = top_features(model, count=3)
    assert (top[top.direction == "negative"].weight < 0).all()
    assert (top[top.direction == "positive"].weight > 0).all()


def test_by_length_buckets_cover_every_review():
    frame = pd.DataFrame({"text": ["ا" * 10, "ا" * 100, "ا" * 200, "ا" * 500] * 2,
                          "label": [0, 1] * 4})
    table = by_length(frame, frame["label"].to_numpy())
    assert table["reviews"].sum() == len(frame)
    assert np.allclose(table["macro_f1"].dropna(), 1.0)


def test_calibrator_keeps_order_and_fixes_a_shift():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.8).astype(int)
    prob = np.where(y == 1, 0.65, 0.35) + rng.normal(0, 0.1, 4000)
    prob = np.clip(prob, 0.01, 0.99)
    fixed = PlattCalibrator().fit(prob, y).transform(prob)
    assert np.array_equal(np.argsort(prob), np.argsort(fixed))
    assert abs(fixed.mean() - y.mean()) < 0.01
