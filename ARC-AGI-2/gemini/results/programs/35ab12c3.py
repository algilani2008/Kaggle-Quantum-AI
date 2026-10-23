# ARC-AGI-2 task 35ab12c3: verified on all train pairs
# model=gemini-4-argon sample=1 round=1 test_correct=1.0

from itertools import permutations
import numpy as np


def get_line_pixels(
    p1: tuple[int, int], p2: tuple[int, int]
) -> list[tuple[int, int]]:
    r1, c1 = p1
    r2, c2 = p2
    steps = max(abs(r2 - r1), abs(c2 - c1))
    if steps == 0:
        return [(r1, c1)]
    dr = (r2 - r1) // steps
    dc = (c2 - c1) // steps
    return [(r1 + i * dr, c1 + i * dc) for i in range(steps + 1)]


def build_shape_pixels(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    n = len(points)
    if n == 1:
        return set(points)

    # Group points along horizontal, vertical, and both 45-degree diagonal lines
    lines = {}
    for idx, (r, c) in enumerate(points):
        for key in [("h", r), ("v", c), ("d1", r + c), ("d2", r - c)]:
            lines.setdefault(key, []).append(idx)

    # Build adjacency graph of collinear consecutive points
    adj = {i: {} for i in range(n)}
    for key, idxs in lines.items():
        if len(idxs) < 2:
            continue
        if key[0] == "h":
            idxs.sort(key=lambda i: points[i][1])
        else:
            idxs.sort(key=lambda i: points[i][0])
        for a, b in zip(idxs, idxs[1:]):
            r1, c1 = points[a]
            r2, c2 = points[b]
            dist = max(abs(r1 - r2), abs(c1 - c2))
            adj[a][b] = dist
            adj[b][a] = dist

    # Find connected components and select valid edges
    visited = set()
    selected_edges = []
    for start in range(n):
        if start in visited:
            continue
        comp = []
        queue = [start]
        visited.add(start)
        while queue:
            curr = queue.pop(0)
            comp.append(curr)
            for nxt in adj[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        max_deg = max(len(adj[u]) for u in comp)
        if max_deg <= 2:
            for u in comp:
                for v in adj[u]:
                    if u < v:
                        selected_edges.append((u, v))
        else:
            # Find the shortest Hamiltonian cycle on the component to eliminate internal chords
            first = comp[0]
            rest = comp[1:]
            best_cost = float("inf")
            best_cycle = None
            for perm in permutations(rest):
                cycle = (first,) + perm
                cost = 0
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if v not in adj[u]:
                        valid = False
                        break
                    cost += adj[u][v]
                if valid and cost < best_cost:
                    best_cost = cost
                    best_cycle = cycle

            if best_cycle is not None:
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    selected_edges.append((u, v))

    shape_pixels = set(points)
    for u, v in selected_edges:
        shape_pixels.update(get_line_pixels(points[u], points[v]))
    return shape_pixels


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    rows, cols = arr.shape
    output = np.zeros_like(arr)

    colors = [int(c) for c in np.unique(arr) if c != 0]
    color_points = {
        c: [tuple(map(int, pt)) for pt in np.argwhere(arr == c)] for c in colors
    }

    multi_colors = [c for c in colors if len(color_points[c]) > 1]
    single_colors = [c for c in colors if len(color_points[c]) == 1]

    # Build shapes for multi-pixel colors
    shapes = {}
    for c in multi_colors:
        shapes[c] = build_shape_pixels(color_points[c])

    # Draw shifted shapes for single-pixel colors first (underneath)
    for c in single_colors:
        sr, sc = color_points[c][0]
        matched = False
        for mc in multi_colors:
            for mr, mc_col in color_points[mc]:
                if max(abs(sr - mr), abs(sc - mc_col)) == 1:
                    dr, dc = sr - mr, sc - mc_col
                    for r, col in shapes[mc]:
                        nr, nc = r + dr, col + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            output[nr, nc] = c
                    matched = True
                    break
            if matched:
                break
        if not matched:
            output[sr, sc] = c

    # Draw main multi-pixel shapes on top
    for c in multi_colors:
        for r, col in shapes[c]:
            output[r, col] = c

    return output.tolist()
