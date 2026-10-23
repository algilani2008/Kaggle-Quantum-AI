# ARC-AGI-2 task 332f06d7: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

from collections import deque
import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    H, W = arr.shape

    zeros = np.argwhere(arr == 0)
    r0, c0 = zeros.min(axis=0)
    r1, c1 = zeros.max(axis=0)
    k = r1 - r0 + 1

    # Free cells are anything other than wall (3)
    free = arr != 3

    # Compute shortest-path distance from every free cell to the target (2)
    dist_to_2 = np.full((H, W), np.inf)
    q = deque()
    for r, c in np.argwhere(arr == 2):
        dist_to_2[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and free[nr, nc]:
                if dist_to_2[nr, nc] == np.inf:
                    dist_to_2[nr, nc] = dist_to_2[r, c] + 1
                    q.append((nr, nc))

    # Precompute valid top-left positions for a k x k block
    valid = np.zeros((H - k + 1, W - k + 1), dtype=bool)
    for r in range(H - k + 1):
        for c in range(W - k + 1):
            valid[r, c] = free[r : r + k, c : c + k].all()

    # BFS to find all k x k positions reachable from (r0, c0)
    visited = np.zeros_like(valid, dtype=bool)
    visited[r0, c0] = True
    q = deque([(r0, c0)])
    reachable = []

    while q:
        r, c = q.popleft()
        reachable.append((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H - k + 1 and 0 <= nc < W - k + 1:
                if valid[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))

    # Pick the reachable position that minimizes the distance to the 2s
    best_r, best_c = min(
        reachable,
        key=lambda pos: dist_to_2[
            pos[0] : pos[0] + k, pos[1] : pos[1] + k
        ].sum(),
    )

    out = arr.copy()
    out[r0 : r0 + k, c0 : c0 + k] = 1
    out[best_r : best_r + k, best_c : best_c + k] = 0
    return out.tolist()
