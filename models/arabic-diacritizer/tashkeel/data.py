"""Download the open Tashkeela-based benchmark (MIT licence) used for training."""
from __future__ import annotations

import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/AliOsm/arabic-text-diacritization/master/dataset/"
SPLITS = ("train", "val", "test")


def load_split(split: str, cache_dir: str | Path = "data") -> list[str]:
    """Return the non-empty lines of one split, downloading it on first use."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{split}.txt"
    if not path.exists():
        urllib.request.urlretrieve(BASE_URL + f"{split}.txt", path)
    return [line.strip() for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
