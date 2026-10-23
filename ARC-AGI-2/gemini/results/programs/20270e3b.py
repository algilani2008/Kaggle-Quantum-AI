# ARC-AGI-2 task 20270e3b: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

from collections import deque


def transform(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])

    def get_components(cells, connectivity=4):
        remaining = set(cells)
        comps = []
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 8:
            dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        while remaining:
            start = next(iter(remaining))
            remaining.remove(start)
            comp = {start}
            q = deque([start])
            while q:
                r, c = q.popleft()
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in remaining:
                        remaining.remove((nr, nc))
                        comp.add((nr, nc))
                        q.append((nr, nc))
            comps.append(comp)
        return comps

    sevens = {(r, c) for r in range(H) for c in range(W) if grid[r][c] == 7}
    non_bg = {(r, c) for r in range(H) for c in range(W) if grid[r][c] != 1}

    s_comps = get_components(sevens, connectivity=4)
    if len(s_comps) != 2:
        s_comps = get_components(sevens, connectivity=8)
    S1, S2 = s_comps[0], s_comps[1]

    pieces = get_components(non_bg, connectivity=4)
    if len(pieces) != 2:
        pieces = get_components(non_bg, connectivity=8)

    P1 = set()
    P2 = set()
    others = []
    for comp in pieces:
        if comp & S1:
            P1 |= comp
        elif comp & S2:
            P2 |= comp
        else:
            others.append(comp)

    for comp in others:
        d1 = min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in comp for r2, c2 in P1)
        d2 = min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in comp for r2, c2 in P2)
        if d1 <= d2:
            P1 |= comp
        else:
            P2 |= comp

    F1 = P1 - S1
    F2 = P2 - S2

    r1_min, c1_min = min(S1)
    r2_min, c2_min = min(S2)

    best_shift = (0, 0)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if all((r + dr, c + dc) in F1 for r, c in S1) and all(
            (r - dr, c - dc) in F2 for r, c in S2
        ):
            shift = (r1_min + dr - r2_min, c1_min + dc - c2_min)
            best_shift = shift
            shifted_F2 = {(r + shift[0], c + shift[1]) for r, c in F2}
            if not (F1 & shifted_F2):
                break

    dr_shift, dc_shift = best_shift
    combined_fours = F1 | {(r + dr_shift, c + dc_shift) for r, c in F2}

    min_r = min(r for r, _ in combined_fours)
    max_r = max(r for r, _ in combined_fours)
    min_c = min(c for _, c in combined_fours)
    max_c = max(c for _, c in combined_fours)

    out_h = max_r - min_r + 1
    out_w = max_c - min_c + 1
    out = [[1] * out_w for _ in range(out_h)]
    for r, c in combined_fours:
        out[r - min_r][c - min_c] = 4

    return out
