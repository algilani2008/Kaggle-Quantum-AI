import pandas as pd
import pytest

from arsent import dataset
from arsent.dataset import deduplicate, read_reviews, to_binary


def frame(rows):
    return pd.DataFrame(rows, columns=["rating", "book_id", "text"])


def test_read_reviews_drops_user_ids_and_keeps_quotes(tmp_path):
    path = tmp_path / "reviews.tsv"
    path.write_text('5\t1\t77\t9\tقال "رائع" جدا\n1\t2\t78\t9\tسيء\n', encoding="utf-8")
    data = read_reviews(path)
    assert "user_id" not in data.columns
    assert data["text"].iloc[0] == 'قال "رائع" جدا'


def test_to_binary_drops_three_stars():
    data = to_binary(frame([(1, 1, "a"), (2, 1, "b"), (3, 1, "c"), (4, 1, "d"), (5, 1, "e")]))
    assert data["label"].tolist() == [0, 0, 1, 1]


def test_deduplicate_matches_after_normalization():
    data = to_binary(frame([(5, 1, "كتاب رائع"), (4, 2, "كتَاب رائع"), (1, 3, "سيء")]))
    clean, report = deduplicate(data)
    assert len(clean) == 2
    assert report["rows_in"] == 3 and report["rows_out"] == 2


def test_deduplicate_removes_conflicting_labels():
    data = to_binary(frame([(5, 1, "عادي"), (1, 2, "عادي"), (5, 3, "ممتاز")]))
    clean, report = deduplicate(data)
    assert clean["text"].tolist() == ["ممتاز"]
    assert report["rows_with_conflicting_labels"] == 2


def test_load_reviews_rejects_changed_file(tmp_path):
    (tmp_path / "reviews.tsv").write_text("5\t1\t2\t3\tنص\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        dataset.load_reviews(tmp_path)
