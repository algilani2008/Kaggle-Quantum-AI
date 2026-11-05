"""Download, verify, load and clean the LABR Arabic book reviews dataset."""

from __future__ import annotations

import csv
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .textnorm import normalize_arabic

BASE_URL = "https://raw.githubusercontent.com/mohamedadaly/LABR/master/data"
REVIEWS_SHA256 = "a72f1b8ece766c918dfdb3c31c2a2da4f408dd2f69a24aef5187c85f313edb04"
COLUMNS = ["rating", "review_id", "user_id", "book_id", "text"]


def download(name: str, data_dir: Path) -> Path:
    """Fetch one file from the LABR repository into ``data_dir`` once."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / name
    if not path.exists():
        temp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", temp)
        temp.rename(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_reviews(path: Path) -> pd.DataFrame:
    """Read reviews.tsv. User ids are dropped: nothing here needs them."""
    frame = pd.read_csv(
        path, sep="\t", header=None, names=COLUMNS,
        quoting=csv.QUOTE_NONE, dtype={"text": str},
    )
    return frame.drop(columns="user_id")


def load_reviews(data_dir: Path) -> pd.DataFrame:
    """Download LABR if needed, check its hash and return all rows."""
    path = download("reviews.tsv", data_dir)
    found = sha256(path)
    if found != REVIEWS_SHA256:
        raise ValueError(f"reviews.tsv hash changed: {found}")
    return read_reviews(path)


def official_split(data_dir: Path, name: str = "2class-unbalanced") -> tuple[np.ndarray, np.ndarray]:
    """Return row indices of the published LABR train and test sets."""
    train = np.loadtxt(download(f"{name}-train.txt", data_dir), dtype=int)
    test = np.loadtxt(download(f"{name}-test.txt", data_dir), dtype=int)
    return train, test


def to_binary(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep 1-2 stars (label 0) and 4-5 stars (label 1); drop 3 stars."""
    kept = frame[frame["rating"] != 3].copy()
    kept["label"] = (kept["rating"] >= 4).astype(int)
    return kept


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep one copy of each normalized text.

    A text that appears with both a positive and a negative rating has no
    single true label, so all of its copies are removed.
    """
    data = frame.copy()
    data["key"] = data["text"].map(normalize_arabic)
    data = data[data["key"].str.len() > 0]
    labels_per_key = data.groupby("key")["label"].transform("nunique")
    conflicting = data[labels_per_key > 1]
    clean = data[labels_per_key == 1].drop_duplicates("key")
    report = {
        "rows_in": len(frame),
        "rows_with_conflicting_labels": len(conflicting),
        "rows_out": len(clean),
    }
    return clean.drop(columns="key").reset_index(drop=True), report
