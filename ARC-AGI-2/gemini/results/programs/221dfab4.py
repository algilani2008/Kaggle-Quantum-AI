# ARC-AGI-2 task 221dfab4: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Rotate the grid so that the edge containing 4s is at the top (row 0)
    rot_k = 0
    for k in range(4):
        rotated = np.rot90(arr, k)
        if np.any(rotated[0] == 4):
            arr = rotated.copy()
            rot_k = k
            break

    beam_cols = np.where(arr[0] == 4)[0]

    for r in range(arr.shape[0]):
        if r % 2 == 1:
            arr[r, beam_cols] = bg
        elif r % 6 == 4:
            arr[r, beam_cols] = 3
            arr[r, arr[r] != bg] = 3
        else:
            arr[r, beam_cols] = 4

    return np.rot90(arr, -rot_k).tolist()
