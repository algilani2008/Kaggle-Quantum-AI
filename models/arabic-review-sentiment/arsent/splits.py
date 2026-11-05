"""Train/validation/test splits that keep every book on one side."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def book_split(groups, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split row positions so that no group lands on both sides."""
    groups = np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(groups, groups=groups))
    return train, test


def random_split(labels, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split on rows; the same book can land on both sides."""
    positions = np.arange(len(labels))
    train, test = train_test_split(
        positions, test_size=test_size, random_state=seed, stratify=labels,
    )
    return train, test


def book_three_way(groups, seed: int, val: float = 0.15, test: float = 0.15):
    """Return train, validation and test positions with disjoint books."""
    groups = np.asarray(groups)
    rest, test_pos = book_split(groups, test, seed)
    inner_train, inner_val = book_split(groups[rest], val / (1 - test), seed + 1)
    return rest[inner_train], rest[inner_val], test_pos
