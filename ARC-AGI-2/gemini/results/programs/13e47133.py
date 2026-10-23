# ARC-AGI-2 task 13e47133: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

from collections import Counter, deque


def transform(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])
    counts = Counter(val for row in grid for val in row)
    (bg, _), (wall, _) = counts.most_common(2)

    out = [row[:] for row in grid]
    visited = [[False] * W for _ in range(H)]

    for r0 in range(H):
        for c0 in range(W):
            if grid[r0][c0] == wall or visited[r0][c0]:
                continue

            # Find 4-connected region of non-wall cells
            comp = []
            q = deque([(r0, c0)])
            visited[r0][c0] = True
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] != wall:
                        visited[nr][nc] = True
                        q.append((nr, nc))

            comp_set = set(comp)
            outside = [
                (r, c)
                for r in range(H)
                for c in range(W)
                if (r, c) not in comp_set
            ]

            # Compute 0-indexed Chebyshev layer for each cell in the region
            layers = {}
            clues = {}
            for r, c in comp:
                dist = min(r + 1, H - r, c + 1, W - c)
                for or_, oc in outside:
                    d = max(abs(r - or_), abs(c - oc))
                    if d < dist:
                        dist = d
                layer = dist - 1
                layers[(r, c)] = layer
                if grid[r][c] != bg:
                    clues[layer] = grid[r][c]

            if not clues:
                continue

            length = max(clues) + 1
            seq = [clues.get(i, bg) for i in range(length)]

            for r, c in comp:
                out[r][c] = seq[layers[(r, c)] % length]

    return out
