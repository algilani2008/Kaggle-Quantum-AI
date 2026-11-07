"""Download, verify and read the MetRec dataset (Al-shaibani, Alyafeai & Ahmad, 2020).

Source: https://github.com/zaidalyafeai/MetRec (MIT License). The archive is
downloaded once into a local folder that is not committed.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://raw.githubusercontent.com/zaidalyafeai/MetRec/master/baits.zip"
SHA256 = "558ac80e54a9111f9d784e898526ce8bc831d06aafd2110cb2acd216a667dbf7"
SPLITS = ("train", "test")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(cache_dir: str | Path) -> Path:
    """Fetch and unpack baits.zip into ``cache_dir`` (once). Returns the folder with train/test."""
    cache = Path(cache_dir)
    target = cache / "final_baits"
    if all((target / f"{s}.txt").exists() for s in SPLITS):
        return target
    cache.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as resp:
        data = resp.read()
    digest = sha256_bytes(data)
    if digest != SHA256:
        raise RuntimeError(f"checksum mismatch for baits.zip: {digest}")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            # only plain .txt files inside final_baits/, never absolute or parent paths
            p = Path(name)
            if p.suffix == ".txt" and p.parent.name == "final_baits" and ".." not in p.parts and not p.is_absolute():
                (target).mkdir(parents=True, exist_ok=True)
                (target / p.name).write_bytes(zf.read(name))
    return target


def parse_line(line: str) -> tuple[int, str] | None:
    """``"<label> <verse>"`` -> (label, verse). Blank or malformed lines give None."""
    line = line.strip()
    if not line:
        return None
    head, _, verse = line.partition(" ")
    if not head.isdigit() or not verse.strip():
        return None
    return int(head), verse.strip()


def read_split(folder: str | Path, split: str) -> tuple[list[str], list[int]]:
    verses, labels = [], []
    with open(Path(folder) / f"{split}.txt", encoding="utf-8") as fh:
        for line in fh:
            parsed = parse_line(line)
            if parsed:
                labels.append(parsed[0])
                verses.append(parsed[1])
    return verses, labels
