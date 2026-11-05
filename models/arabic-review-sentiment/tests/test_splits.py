import numpy as np

from arsent.splits import book_split, book_three_way, random_split


def test_book_split_keeps_books_apart():
    groups = np.repeat(np.arange(50), 4)
    train, test = book_split(groups, 0.2, seed=1)
    assert not set(groups[train]) & set(groups[test])


def test_three_way_covers_all_rows_without_shared_books():
    groups = np.repeat(np.arange(100), 3)
    train, val, test = book_three_way(groups, seed=3)
    assert len(set(train) | set(val) | set(test)) == len(groups)
    books = [set(groups[p]) for p in (train, val, test)]
    assert not (books[0] & books[1] or books[0] & books[2] or books[1] & books[2])


def test_three_way_is_reproducible():
    groups = np.repeat(np.arange(40), 2)
    first = book_three_way(groups, seed=9)
    second = book_three_way(groups, seed=9)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_random_split_keeps_class_balance():
    labels = np.array([0] * 200 + [1] * 800)
    _, test = random_split(labels, 0.2, seed=2)
    assert abs(labels[test].mean() - 0.8) < 0.01
