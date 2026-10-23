# ARC-AGI-2 task 3dc255db: verified on all train pairs
# model=gemini-4-argon sample=1 round=1 test_correct=1.0

from collections import deque


def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])
    out = [row[:] for row in grid]

    visited = [[False] * W for _ in range(H)]
    components = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                q = deque([(r, c)])
                visited[r][c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                rmin = min(cr for cr, _ in cells)
                rmax = max(cr for cr, _ in cells)
                cmin = min(cc for _, cc in cells)
                cmax = max(cc for _, cc in cells)
                components.append((color, set(cells), (rmin, rmax, cmin, cmax)))

    for i, (color, cells, (rmin, rmax, cmin, cmax)) in enumerate(components):
        if len(cells) < 5 or (rmax - rmin + 1) < 3 or (cmax - cmin + 1) < 3:
            continue
        # Ensure this component's bounding box is not strictly inside another component's bounding box
        if any(
            j != i and br0 <= rmin and rmax <= br1 and bc0 <= cmin and cmax <= bc1
            and (br0, br1, bc0, bc1) != (rmin, rmax, cmin, cmax)
            for j, (_, _, (br0, br1, bc0, bc1)) in enumerate(components)
        ):
            continue

        inner = [
            (r, c)
            for r in range(rmin, rmax + 1)
            for c in range(cmin, cmax + 1)
            if grid[r][c] != 0 and grid[r][c] != color
        ]
        if not inner:
            continue

        dust_color = grid[inner[0][0]][inner[0][1]]
        count = len(inner)
        for r, c in inner:
            out[r][c] = 0

        # Identify the nozzle tip and emission direction
        tip = None
        direction = None

        top_cells = [c for r, c in cells if r == rmin]
        if len(top_cells) == 1 and cmin < top_cells[0] < cmax:
            c0 = top_cells[0]
            col_rows = sorted(r for r, c in cells if c == c0)
            if col_rows == list(range(rmin, rmin + len(col_rows))):
                tip = (rmin, c0)
                direction = (-1, 0)

        bottom_cells = [c for r, c in cells if r == rmax]
        if tip is None and len(bottom_cells) == 1 and cmin < bottom_cells[0] < cmax:
            c0 = bottom_cells[0]
            col_rows = sorted(r for r, c in cells if c == c0)
            if col_rows == list(range(rmax - len(col_rows) + 1, rmax + 1)):
                tip = (rmax, c0)
                direction = (1, 0)

        left_cells = [r for r, c in cells if c == cmin]
        if tip is None and len(left_cells) == 1 and rmin < left_cells[0] < rmax:
            r0 = left_cells[0]
            row_cols = sorted(c for r, c in cells if r == r0)
            if row_cols == list(range(cmin, cmin + len(row_cols))):
                tip = (r0, cmin)
                direction = (0, -1)

        right_cells = [r for r, c in cells if c == cmax]
        if tip is None and len(right_cells) == 1 and rmin < right_cells[0] < rmax:
            r0 = right_cells[0]
            row_cols = sorted(c for r, c in cells if r == r0)
            if row_cols == list(range(cmax - len(row_cols) + 1, cmax + 1)):
                tip = (r0, cmax)
                direction = (0, 1)

        if tip is not None and direction is not None:
            tr, tc = tip
            dr, dc = direction
            for k in range(1, count + 1):
                nr, nc = tr + k * dr, tc + k * dc
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr][nc] = dust_color

    return out
