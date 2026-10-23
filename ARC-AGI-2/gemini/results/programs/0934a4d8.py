# ARC-AGI-2 task 0934a4d8: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    H, W = arr.shape

    # Find the vertical symmetry sum R_sum (r + r' = R_sum)
    best_r_sum = H - 1
    best_r_score = (-1.0, -1)
    for r_sum in range(H // 2, 2 * H - H // 2):
        matches = 0
        total = 0
        for r in range(H):
            r2 = r_sum - r
            if 0 <= r2 < H:
                mask = (arr[r, :] != 8) & (arr[r2, :] != 8)
                total += int(np.sum(mask))
                matches += int(np.sum(arr[r, mask] == arr[r2, mask]))
        if total > 0:
            score = (matches / total, matches)
            if score > best_r_score:
                best_r_score = score
                best_r_sum = r_sum

    # Find the horizontal symmetry sum C_sum (c + c' = C_sum)
    best_c_sum = W - 1
    best_c_score = (-1.0, -1)
    for c_sum in range(W // 2, 2 * W - W // 2):
        matches = 0
        total = 0
        for c in range(W):
            c2 = c_sum - c
            if 0 <= c2 < W:
                mask = (arr[:, c] != 8) & (arr[:, c2] != 8)
                total += int(np.sum(mask))
                matches += int(np.sum(arr[mask, c] == arr[mask, c2]))
        if total > 0:
            score = (matches / total, matches)
            if score > best_c_score:
                best_c_score = score
                best_c_sum = c_sum

    # Locate the rectangular region of 8s
    rows, cols = np.where(arr == 8)
    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())

    out = []
    for r in range(r_min, r_max + 1):
        row_vals = []
        for c in range(c_min, c_max + 1):
            val = arr[r, c]
            if val == 8:
                candidates = [
                    (best_r_sum - r, c),
                    (r, best_c_sum - c),
                    (best_r_sum - r, best_c_sum - c),
                    (c, r),
                    (c, best_r_sum - r),
                    (best_c_sum - c, r),
                    (best_c_sum - c, best_r_sum - r),
                ]
                for nr, nc in candidates:
                    if 0 <= nr < H and 0 <= nc < W and arr[nr, nc] != 8:
                        val = arr[nr, nc]
                        break
            row_vals.append(int(val))
        out.append(row_vals)

    return out
