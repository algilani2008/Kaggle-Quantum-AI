# ARC-AGI-2 task 1818057f: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    out = arr.copy()
    h, w = arr.shape

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (
                arr[r, c] == 4
                and arr[r - 1, c] == 4
                and arr[r + 1, c] == 4
                and arr[r, c - 1] == 4
                and arr[r, c + 1] == 4
            ):
                out[r, c] = 8
                out[r - 1, c] = 8
                out[r + 1, c] = 8
                out[r, c - 1] = 8
                out[r, c + 1] = 8

    return out.tolist()
