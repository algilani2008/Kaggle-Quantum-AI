# ARC-AGI-2 task 31f7f899: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    H, W = arr.shape
    bg_color = 8
    spine_color = 6

    # Find the central horizontal spine row (the row containing spine_color)
    mid_r = H // 2
    for r in range(H):
        if spine_color in arr[r]:
            mid_r = r
            break

    # Group contiguous columns of the same non-spine color into bars
    bars = []
    c = 0
    while c < W:
        color = int(arr[mid_r, c])
        if color != spine_color and color != bg_color:
            c_start = c
            height = int(np.sum(arr[:, c] != bg_color))
            while (
                c + 1 < W
                and arr[mid_r, c + 1] == color
                and int(np.sum(arr[:, c + 1] != bg_color)) == height
            ):
                c += 1
            bars.append((c_start, c, color, height))
        c += 1

    # Sort the heights of the bars in ascending order from left to right
    sorted_heights = sorted(b[3] for b in bars)

    # Build the output grid
    out = np.full((H, W), bg_color, dtype=int)
    out[mid_r, :] = arr[mid_r, :]

    for (c_start, c_end, color, _), new_h in zip(bars, sorted_heights):
        half = new_h // 2
        out[mid_r - half : mid_r + half + 1, c_start : c_end + 1] = color

    return out.tolist()
