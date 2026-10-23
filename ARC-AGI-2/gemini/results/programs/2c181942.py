# ARC-AGI-2 task 2c181942: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    rows, cols = arr.shape
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # 1. Locate the 4x4 multi-colored central ring
    r0 = c0 = -1
    for r in range(rows - 3):
        for c in range(cols - 3):
            sub = arr[r:r+4, c:c+4]
            top = sub[0, 1]
            bot = sub[3, 1]
            left = sub[1, 0]
            right = sub[1, 3]
            if (top != bg and sub[0, 2] == top and
                bot != bg and sub[3, 2] == bot and
                left != bg and sub[2, 0] == left and
                right != bg and sub[2, 3] == right):
                empty_spots = [
                    sub[0, 0], sub[0, 3], sub[3, 0], sub[3, 3],
                    sub[1, 1], sub[1, 2], sub[2, 1], sub[2, 2]
                ]
                if all(x == bg for x in empty_spots) and len({top, bot, left, right}) > 1:
                    r0, c0 = r, c
                    break
        if r0 != -1:
            break

    out = np.full_like(arr, bg)
    out[r0:r0+4, c0:c0+4] = arr[r0:r0+4, c0:c0+4]

    # Mask out the central ring to find the scattered shapes
    work = arr.copy()
    work[r0:r0+4, c0:c0+4] = bg

    def is_connector(edge: np.ndarray, color: int) -> bool:
        n = len(edge)
        if n % 2 != 0:
            return False
        expected = np.full(n, bg)
        expected[n // 2 - 1 : n // 2 + 1] = color
        return np.array_equal(edge, expected)

    sides = [
        ("top", int(arr[r0, c0 + 1])),
        ("bot", int(arr[r0 + 3, c0 + 1])),
        ("left", int(arr[r0 + 1, c0])),
        ("right", int(arr[r0 + 1, c0 + 3])),
    ]

    # 2. Rotate and attach each matching shape to its corresponding ring side
    for direction, color in sides:
        pts = np.argwhere(work == color)
        if len(pts) == 0:
            continue
        rmin, cmin = pts.min(axis=0)
        rmax, cmax = pts.max(axis=0)
        shape = np.where(work[rmin:rmax+1, cmin:cmax+1] == color, color, bg)

        for k in range(4):
            rot = np.rot90(shape, k)
            h, w = rot.shape
            if direction == "top" and is_connector(rot[-1, :], color):
                sr, sc = r0 - h, c0 + 1 - (w // 2 - 1)
                break
            if direction == "bot" and is_connector(rot[0, :], color):
                sr, sc = r0 + 4, c0 + 1 - (w // 2 - 1)
                break
            if direction == "left" and is_connector(rot[:, -1], color):
                sr, sc = r0 + 1 - (h // 2 - 1), c0 - w
                break
            if direction == "right" and is_connector(rot[:, 0], color):
                sr, sc = r0 + 1 - (h // 2 - 1), c0 + 4
                break

        mask = rot == color
        out[sr:sr+h, sc:sc+w][mask] = color

    return out.tolist()
