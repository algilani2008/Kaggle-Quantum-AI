# ARC-AGI-2 task 16de56c4: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)

    # Rotate the grid so that the starting border is on the left (column 0)
    rotations = 0
    for k in range(4):
        rotated = np.rot90(arr, k)
        non_zero_rows = [r for r in range(rotated.shape[0]) if np.any(rotated[r] != 0)]
        if non_zero_rows and all(rotated[r, 0] != 0 for r in non_zero_rows):
            arr = rotated
            rotations = k
            break

    h, w = arr.shape
    out = arr.copy()

    for r in range(h):
        start_color = arr[r, 0]
        if start_color == 0:
            continue

        # Find step size d from the next pixel matching the start color
        matches = [c for c in range(1, w) if arr[r, c] == start_color]
        if not matches:
            continue
        d = matches[0]

        # Trace the arithmetic progression 0, d, 2d, ...
        positions = list(range(0, w, d))
        fill_color = start_color
        stop_idx = len(positions)

        for i, c in enumerate(positions):
            if arr[r, c] != 0 and arr[r, c] != start_color:
                fill_color = arr[r, c]
                stop_idx = i + 1
                break

        for c in positions[:stop_idx]:
            out[r, c] = fill_color

    # Rotate back to the original orientation
    out = np.rot90(out, -rotations)
    return out.tolist()
