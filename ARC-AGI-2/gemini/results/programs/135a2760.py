# ARC-AGI-2 task 135a2760: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

from collections import Counter
import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    H, W = arr.shape
    bg = arr[0, 0]
    out = arr.copy()

    # Find all minimal hollow rectangular frames of uniform non-background colour
    frames = []
    for r1 in range(H - 2):
        for c1 in range(W - 2):
            fc = arr[r1, c1]
            if fc == bg:
                continue
            for r2 in range(r1 + 2, H):
                if arr[r2, c1] != fc:
                    break
                for c2 in range(c1 + 2, W):
                    if arr[r1, c2] != fc:
                        break
                    if (
                        np.all(arr[r1, c1 : c2 + 1] == fc)
                        and np.all(arr[r2, c1 : c2 + 1] == fc)
                        and np.all(arr[r1 : r2 + 1, c1] == fc)
                        and np.all(arr[r1 : r2 + 1, c2] == fc)
                    ):
                        # Ensure no interior row or column is completely frame-coloured
                        sub = arr[r1 + 1 : r2, c1 + 1 : c2]
                        if not np.any(np.all(sub == fc, axis=1)) and not np.any(
                            np.all(sub == fc, axis=0)
                        ):
                            frames.append((r1 + 1, c1 + 1, r2 - 1, c2 - 1))

    for ir1, ic1, ir2, ic2 in frames:
        sub = arr[ir1 : ir2 + 1, ic1 : ic2 + 1]
        sh, sw = sub.shape
        transposed = sw > sh
        if transposed:
            sub = sub.T  # shape: (L, D) where L is the long axis
        L, D = sub.shape

        best_score = float("inf")
        best_recon = sub

        max_p = max(1, L // 3)
        for p in range(1, max_p + 1):
            recon = np.empty_like(sub)
            mismatches = 0
            for rem in range(p):
                indices = range(rem, L, p)
                for d in range(D):
                    vals = [sub[i, d] for i in indices]
                    mode = Counter(vals).most_common(1)[0][0]
                    for i in indices:
                        recon[i, d] = mode
                        if sub[i, d] != mode:
                            mismatches += 1
            score = mismatches + 0.4 * p
            if score < best_score:
                best_score = score
                best_recon = recon

        if transposed:
            best_recon = best_recon.T
        out[ir1 : ir2 + 1, ic1 : ic2 + 1] = best_recon

    return out.tolist()
