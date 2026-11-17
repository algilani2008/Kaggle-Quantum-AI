import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tashkeel import Diacritizer, join_text, split_text, strip_diacritics  # noqa: E402
from tashkeel.baselines import LetterMajority, WordLookup  # noqa: E402
from tashkeel.features import featurize  # noqa: E402
from tashkeel.metrics import evaluate  # noqa: E402
from tashkeel.model import lines_to_xy  # noqa: E402
from tashkeel.text import canonical, FATHA, SHADDA  # noqa: E402

RAW = [
    "ذَهَبَ الطَّالِبُ إلَى الْمَدْرَسَةِ",
    "قَالَ الشَّيْخُ : الْعِلْمُ نُورٌ",
    "كَتَبَ الطَّالِبُ الدَّرْسَ",
    "رَأَيْتُ رَجُلًا كَرِيمًا",
] * 5
# store diacritics in canonical order (shadda first) so round trips are exact
SAMPLE = [join_text(*split_text(s)) for s in RAW]


def test_split_join_roundtrip():
    for line in SAMPLE:
        bare, labels = split_text(line)
        assert bare == strip_diacritics(line)
        assert join_text(bare, labels) == line
        assert len(bare) == len(labels)


def test_canonical_puts_shadda_first():
    assert canonical(FATHA + SHADDA) == SHADDA + FATHA
    assert canonical(FATHA + FATHA) == FATHA


def test_non_letters_never_get_labels():
    bare, labels = split_text("َ 12 ،")  # a stray diacritic at the start is dropped
    assert bare == " 12 ،"
    assert all(lab == "" for lab in labels)


def test_featurize_shapes_and_mapping():
    lines = [strip_diacritics(s) for s in SAMPLE[:3]]
    X, li, ci = featurize(lines, n_bits=12)
    n_letters = sum(ch != " " and ch != ":" for line in lines for ch in line)
    assert X.shape == (n_letters, 1 << 12)
    for a, b in zip(li, ci):
        assert lines[a][b] not in " :"
    # featurisation is deterministic
    X2, _, _ = featurize(lines, n_bits=12)
    assert (X != X2).nnz == 0


def test_metrics_perfect_and_wrong():
    bare, gold = zip(*(split_text(s) for s in SAMPLE[:2]))
    m = evaluate(bare, gold, gold)
    assert m["DER"] == 0 and m["WER"] == 0
    wrong = [[FATHA] * len(g) for g in gold]
    m2 = evaluate(bare, gold, wrong)
    assert 0 < m2["DER"] <= 1 and m2["WER"] > 0


def test_baselines_memorise_training_words():
    wl = WordLookup().fit(SAMPLE)
    bare, gold = zip(*(split_text(s) for s in SAMPLE[:4]))
    assert evaluate(bare, gold, wl.predict_labels(list(bare)))["WER"] == 0
    lm = LetterMajority().fit(SAMPLE)
    assert len(lm.predict_labels(["كتب"])[0]) == 3


def _tiny_model():
    from sklearn.linear_model import SGDClassifier

    classes = sorted({lab for s in SAMPLE for lab in split_text(s)[1]})
    cidx = {c: i for i, c in enumerate(classes)}
    X, y = lines_to_xy(SAMPLE, cidx, 14)
    clf = SGDClassifier(loss="hinge", alpha=1e-6, max_iter=50, random_state=0).fit(X, y)
    return Diacritizer(classes, clf.coef_.astype(np.float32), clf.intercept_, 14)


def test_tiny_model_learns_and_roundtrips(tmp_path):
    model = _tiny_model()
    out = model.diacritize(strip_diacritics(SAMPLE[0]))
    assert strip_diacritics(out) == strip_diacritics(SAMPLE[0])  # letters untouched
    bare, gold = zip(*(split_text(s) for s in SAMPLE[:4]))
    assert evaluate(bare, gold, model.predict_labels(list(bare)))["DER"] < 0.1

    model.save(tmp_path / "m")
    loaded = Diacritizer.load(tmp_path / "m")
    assert loaded.classes == model.classes
    assert loaded.diacritize(strip_diacritics(SAMPLE[1])).count(" ") == SAMPLE[1].count(" ")


def test_tanween_only_on_last_letter():
    model = _tiny_model()
    line = "رأيت رجلا كريما مستقبلا"
    labels = model.predict_labels([line])[0]
    for i, lab in enumerate(labels):
        if any(t in lab for t in "\u064b\u064c\u064d"):
            rest = line[i + 1 : i + 3]
            assert len(rest) < 2 or " " in rest


def test_existing_diacritics_are_replaced_and_empty_input():
    model = _tiny_model()
    assert model.diacritize("") == ""
    assert model.diacritize("123 !") == "123 !"
    assert strip_diacritics(model.diacritize(SAMPLE[2])) == strip_diacritics(SAMPLE[2])


RELEASED = ROOT / "weights" / "tashkeel-linear.npz"


@pytest.mark.skipif(not RELEASED.exists(), reason="released weights not built yet")
def test_released_model_on_common_sentence():
    model = Diacritizer.load(ROOT / "weights" / "tashkeel-linear")
    out = model.diacritize("قال رسول الله")
    assert strip_diacritics(out) == "قال رسول الله"
    assert sum(ch in "َُِّْ" for ch in out) >= 5
