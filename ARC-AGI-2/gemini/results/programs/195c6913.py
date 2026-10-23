# ARC-AGI-2 task 195c6913: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

from collections import Counter
import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    H, W = arr.shape
    out = arr.copy()

    # The two most common colours are wall_color and path_color; (0, 0) is wall_color.
    counts = Counter(arr.flatten())
    top2 = [c for c, _ in counts.most_common(2)]
    wall_color = int(arr[0, 0])
    path_color = top2[1] if top2[0] == wall_color else top2[0]

    # Find 2x2 legend blocks (top-left sequence blocks and bottom-right collision block).
    top_left_blocks = []
    hit_color = None
    visited = np.zeros((H, W), dtype=bool)

    for r in range(H - 1):
        for c in range(W - 1):
            val = int(arr[r, c])
            if val not in (wall_color, path_color) and not visited[r, c]:
                if (
                    arr[r, c + 1] == val
                    and arr[r + 1, c] == val
                    and arr[r + 1, c + 1] == val
                ):
                    visited[r : r + 2, c : c + 2] = True
                    out[r : r + 2, c : c + 2] = wall_color
                    if r < H // 2 and c < W // 2:
                        top_left_blocks.append((c, val))
                    else:
                        hit_color = val

    top_left_blocks.sort(key=lambda x: x[0])
    seq = [val for _, val in top_left_blocks]

    # Find starting pixels in column 0 and trace each path.
    for r0 in range(H):
        if arr[r0, 0] not in (wall_color, path_color) and not visited[r0, 0]:
            r, c = r0, 0
            dr, dc = 0, 1
            idx = 0
            out[r, c] = seq[idx % len(seq)]

            while True:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if arr[nr, nc] == path_color:
                    r, c = nr, nc
                    idx += 1
                    out[r, c] = seq[idx % len(seq)]
                else:
                    out[nr, nc] = hit_color
                    dr, dc = (-1, 0) if (dr, dc) == (0, 1) else (0, 1)
                    if not (
                        0 <= r + dr < H
                        and 0 <= c + dc < W
                        and arr[r + dr, c + dc] == path_color
                    ):
                        break

    return out.tolist()
