# ARC-AGI-2 task 409aa875: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

from collections import Counter


def transform(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Determine background color (most frequent color)
    bg = Counter(val for row in grid for val in row).most_common(1)[0][0]

    # Find 8-connected components of non-background cells
    visited = [[False] * W for _ in range(H)]
    components = []
    cell_to_comp = {}

    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not visited[r][c]:
                comp = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for nr in range(max(0, cr - 1), min(H, cr + 2)):
                        for nc in range(max(0, cc - 1), min(W, cc + 2)):
                            if not visited[nr][nc] and grid[nr][nc] != bg:
                                visited[nr][nc] = True
                                stack.append((nr, nc))
                comp_idx = len(components)
                components.append(comp)
                for cell in comp:
                    cell_to_comp[cell] = comp_idx

    # Determine the vertex, direction, and target (5 steps away) for each arrow
    hits = Counter()
    for comp in components:
        vertex = min(
            comp,
            key=lambda p: sum((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in comp),
        )
        sum_dr = sum(vertex[0] - q[0] for q in comp)
        sum_dc = sum(vertex[1] - q[1] for q in comp)
        dr = (sum_dr > 0) - (sum_dr < 0)
        dc = (sum_dc > 0) - (sum_dc < 0)

        target = (vertex[0] + 5 * dr, vertex[1] + 5 * dc)
        if 0 <= target[0] < H and 0 <= target[1] < W:
            hits[target] += 1

    # Apply target hits to the output grid
    for (tr, tc), count in hits.items():
        color = 9 if count == 1 else 1
        if (tr, tc) in cell_to_comp:
            for cr, cc in components[cell_to_comp[(tr, tc)]]:
                out[cr][cc] = color
        else:
            out[tr][tc] = color

    return out
