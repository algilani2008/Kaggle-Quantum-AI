import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bahr.data import parse_line, sha256_bytes  # noqa: E402
from bahr.features import (  # noqa: E402
    diacritic_density,
    halves,
    letter_counts,
    marked_skeleton,
    rhythm_skeleton,
    strip_diacritics,
)
from bahr.labels import METERS, METERS_AR  # noqa: E402
from bahr.metrics import accuracy, bootstrap_ci, confusion, macro_f1, per_class_report  # noqa: E402
from bahr.model import MeterClassifier  # noqa: E402


# ---------- features ----------
def test_halves_split_and_fallback():
    assert halves("أ ب # ج د") == ["أ ب", "ج د"]
    assert halves("بلا فاصل") == ["بلا فاصل", ""]


def test_rhythm_skeleton_symbols():
    # كَتَبَ -> C v C v C v ; كِتَابٌ -> C v C v L C n
    assert rhythm_skeleton("كَتَبَ") == "CvCvCv"
    assert rhythm_skeleton("كِتَابٌ") == "CvCvLCn"
    assert rhythm_skeleton("\u0645\u064f\u062f\u0651\u064e") == "CvCdv"  # mudda: shadda then fatha
    assert rhythm_skeleton("قُمْ") == "CvCs"
    assert rhythm_skeleton("  هل  أنت ") == "CC_CCC"


def test_marked_skeleton_has_boundaries():
    s = marked_skeleton("كَتَبَ # قُمْ")
    assert s == "^CvCvCv$ ^CvCs$"


def test_strip_diacritics_and_density():
    assert strip_diacritics("كَتَبَ") == "كتب"
    assert diacritic_density("كتب") == 0.0
    assert diacritic_density("كَتَبَ") == 1.0
    assert diacritic_density("") == 0.0


def test_letter_counts_capped():
    assert letter_counts("كتب # قم") == [3, 2]
    assert letter_counts("ب" * 100 + " # ب", cap=40) == [40, 1]


# ---------- data ----------
def test_parse_line():
    assert parse_line("3 قفا نبك # من ذكرى\n") == (3, "قفا نبك # من ذكرى")
    assert parse_line("") is None
    assert parse_line("x not a label") is None
    assert parse_line("5") is None


def test_sha256_known_value():
    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_labels_complete():
    assert len(METERS) == 14 and len(set(METERS)) == 14
    assert set(METERS_AR) == set(METERS)


# ---------- metrics ----------
def test_metrics_basic():
    y = [0, 0, 1, 1]
    p = [0, 1, 1, 1]
    assert accuracy(y, p) == 0.75
    assert confusion(y, p, 2).tolist() == [[1, 1], [0, 2]]
    rep = per_class_report(y, p, 2)
    assert rep[0]["precision"] == 1.0 and rep[0]["recall"] == 0.5
    assert macro_f1(y, y) == 1.0
    lo, hi = bootstrap_ci(y, p, n_boot=200)
    assert 0.0 <= lo <= 0.75 <= hi <= 1.0
    with pytest.raises(ValueError):
        accuracy([], [])


# ---------- model ----------
def _toy_data():
    # Two clearly different synthetic "rhythms" (synthetic test data, not poetry)
    a = ["كَتَبَ كَتَبَ # كَتَبَ كَتَبَ", "سَمِعَ سَمِعَ # سَمِعَ سَمِعَ", "فَعَلَ فَعَلَ # فَعَلَ فَعَلَ"] * 4
    b = ["قُمْ قُمْ قُمْ # قُمْ قُمْ", "مِنْ عِنْدِ # مِنْ عِنْدِ", "لَمْ يَكُنْ # لَمْ يَكُنْ"] * 4
    return a + b, [0] * len(a) + [1] * len(b)


def _small_settings():
    return {
        "text": {"ngram_range": (1, 3), "max_features": 500, "min_df": 1},
        "rhythm": {"ngram_range": (2, 4), "max_features": 500, "min_df": 1},
    }


def test_fit_predict_and_topk():
    X, y = _toy_data()
    m = MeterClassifier(C=1.0, block_settings=_small_settings()).fit(X, y)
    assert accuracy(y, m.predict(X)) == 1.0
    top = m.top_k(X[0], k=2)
    assert top[0]["label"] == 0 and top[0]["score"] >= top[1]["score"]
    assert top[0]["meter"] == METERS[0]


def test_save_load_roundtrip_without_pickle(tmp_path):
    X, y = _toy_data()
    m = MeterClassifier(C=1.0, block_settings=_small_settings()).fit(X, y)
    m.save(tmp_path, half_precision=False)
    r = MeterClassifier.load(tmp_path)
    np.testing.assert_allclose(r.decision_function(X), m.decision_function(X), rtol=1e-5, atol=1e-5)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json", "vocab_rhythm.json", "vocab_text.json", "weights.npz"]


def test_bad_blocks_and_untrained():
    with pytest.raises(ValueError):
        MeterClassifier(blocks=("nope",))
    with pytest.raises(RuntimeError):
        MeterClassifier().predict(["كتب # كتب"])


def test_released_weights_if_present():
    w = ROOT / "weights"
    if not (w / "config.json").exists():
        pytest.skip("weights not built yet; run train.py")
    m = MeterClassifier.load(w)
    # first line of the Mu'allaqa of Imru' al-Qais, classically in al-Taweel
    out = m.top_k("قفا نبك من ذكرى حبيب ومنزل # بسقط اللوى بين الدخول فحومل", k=3)
    assert len(out) == 3 and all(c["meter"] in METERS for c in out)
