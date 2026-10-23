# ARC-AGI-2 task 31f7f899: verified on all train pairs
# model=gemini-4-argon sample=2 round=0 test_correct=1.0

import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    bg = 8
    axis_color = 6

    # Find the horizontal axis row (the row containing the axis color 6)
    axis_rows = np.where((arr == axis_color).any(axis=1))[0]
    r_axis = int(axis_rows[0])

    # Identify each vertical bar as a contiguous run of columns on the axis row
    # with the same non-axis color and the same vertical span.
    bars = []
    cols = arr.shape[1]
    c = 0
    while c < cols:
        color = int(arr[r_axis, c])
        if color == axis_color or color == bg:
            c += 1
            continue

        non_bg_rows = np.where(arr[:, c] != bg)[0]
        span = (int(non_bg_rows[0]), int(non_bg_rows[-1]))

        c_end = c
        while c_end + 1 < cols and int(arr[r_axis, c_end + 1]) == color:
            next_rows = np.where(arr[:, c_end + 1] != bg)[0]
            next_span = (int(next_rows[0]), int(next_rows[-1]))
            if next_span == span:
                c_end += 1
            else:
                break

        bars.append((c, c_end, color, span))
        c = c_end + 1

    # Sort the vertical spans by total height (ascending)
    sorted_spans = sorted([b[3] for b in bars], key=lambda s: s[1] - s[0])

    # Construct the output grid
    out = np.full_like(arr, bg)
    out[r_axis, :] = arr[r_axis, :]

    for (c_start, c_end, color, _), (r_start, r_end) in zip(bars, sorted_spans):
        out[r_start : r_end + 1, c_start : c_end + 1] = color

    return out.tolist()
