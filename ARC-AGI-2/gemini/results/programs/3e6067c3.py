# ARC-AGI-2 task 3e6067c3: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

from collections import Counter


def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])
    out = [row[:] for row in grid]

    # Background color is the most frequent color in the grid
    counts = Counter(val for row in grid for val in row)
    bg_color = counts.most_common(1)[0][0]

    # Find connected components of non-background cells
    visited = [[False] * W for _ in range(H)]
    boxes = []
    seq_pixels = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg_color and not visited[r][c]:
                stack = [(r, c)]
                visited[r][c] = True
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] != bg_color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))

                if len(cells) == 1:
                    seq_pixels.append(cells[0])
                else:
                    r1 = min(cr for cr, _ in cells)
                    r2 = max(cr for cr, _ in cells)
                    c1 = min(cc for _, cc in cells)
                    c2 = max(cc for _, cc in cells)
                    frame_color = grid[r1][c1]
                    inner_cells = [
                        (cr, cc)
                        for cr in range(r1, r2 + 1)
                        for cc in range(c1, c2 + 1)
                        if grid[cr][cc] != frame_color
                    ]
                    ir1 = min(cr for cr, _ in inner_cells)
                    ir2 = max(cr for cr, _ in inner_cells)
                    ic1 = min(cc for _, cc in inner_cells)
                    ic2 = max(cc for _, cc in inner_cells)
                    center_color = grid[ir1][ic1]
                    boxes.append({
                        "r1": r1, "r2": r2, "c1": c1, "c2": c2,
                        "ir1": ir1, "ir2": ir2, "ic1": ic1, "ic2": ic2,
                        "color": center_color,
                    })

    # Order the bottom sequence pixels from left to right
    seq_pixels.sort(key=lambda pos: pos[1])
    sequence = [grid[r][c] for r, c in seq_pixels]

    # Build adjacency list between aligned boxes with no other box in between
    n_boxes = len(boxes)
    adj = [[] for _ in range(n_boxes)]
    for i in range(n_boxes):
        for j in range(i + 1, n_boxes):
            A, B = boxes[i], boxes[j]
            if A["r1"] == B["r1"] and A["r2"] == B["r2"]:
                c_start = min(A["c2"], B["c2"]) + 1
                c_end = max(A["c1"], B["c1"]) - 1
                if all(grid[A["r1"]][c] == bg_color for c in range(c_start, c_end + 1)):
                    adj[i].append(j)
                    adj[j].append(i)
            elif A["c1"] == B["c1"] and A["c2"] == B["c2"]:
                r_start = min(A["r2"], B["r2"]) + 1
                r_end = max(A["r1"], B["r1"]) - 1
                if all(grid[r][A["c1"]] == bg_color for r in range(r_start, r_end + 1)):
                    adj[i].append(j)
                    adj[j].append(i)

    # Find the path of boxes matching the color sequence
    path = []
    used = [False] * n_boxes

    def dfs(idx: int) -> bool:
        if idx == len(sequence):
            return True
        target_color = sequence[idx]
        candidates = range(n_boxes) if idx == 0 else adj[path[-1]]
        for nxt in candidates:
            if not used[nxt] and boxes[nxt]["color"] == target_color:
                used[nxt] = True
                path.append(nxt)
                if dfs(idx + 1):
                    return True
                path.pop()
                used[nxt] = False
        return False

    dfs(0)

    # Draw bridges between consecutive boxes in the path
    for k in range(len(path) - 1):
        A = boxes[path[k]]
        B = boxes[path[k + 1]]
        color = A["color"]
        if A["r1"] == B["r1"]:
            c_start = min(A["c2"], B["c2"]) + 1
            c_end = max(A["c1"], B["c1"]) - 1
            for r in range(A["ir1"], A["ir2"] + 1):
                for c in range(c_start, c_end + 1):
                    out[r][c] = color
        else:
            r_start = min(A["r2"], B["r2"]) + 1
            r_end = max(A["r1"], B["r1"]) - 1
            for r in range(r_start, r_end + 1):
                for c in range(A["ic1"], A["ic2"] + 1):
                    out[r][c] = color

    return out
