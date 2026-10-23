# ARC-AGI-2 task 4c3d4a41: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    out = arr.copy()
    h, w = arr.shape

    # Locate the lock (the frame that touches the top row) and the key on the left
    lock_cols = np.where(arr[0] == 5)[0]
    c_lock = int(lock_cols[0])

    key_mask = np.zeros_like(arr, dtype=bool)
    key_mask[:, :c_lock] = arr[:, :c_lock] == 5
    key_rows, key_cols = np.where(key_mask)

    c_key_start = int(key_cols.min())
    c_key_end = int(key_cols.max())
    r_base = int(key_rows.max())

    # Remove the key from its initial position
    out[key_mask] = 0

    # Horizontal shift aligns the first key prong with the first pin column inside the lock
    shift = (c_lock + 2) - c_key_start

    for c_k in range(c_key_start, c_key_end + 1):
        c_p = c_k + shift
        prong_rows = np.where(arr[:, c_k] == 5)[0]
        if len(prong_rows) == 0:
            continue
        t = int(prong_rows.min())

        # Check for an existing pin in the target lock column (rows 1 .. r_base)
        pin_rows = np.where(arr[1 : r_base + 1, c_p] != 0)[0] + 1
        if len(pin_rows) > 0:
            b = int(pin_rows.max())
            up_shift = max(0, b - t + 1)
            for r in range(1, t):
                src_r = r + up_shift
                out[r, c_p] = arr[src_r, c_p] if src_r <= b else 0

        # Place the key column in the lock
        out[t : r_base + 1, c_p] = 5

    return out.tolist()
