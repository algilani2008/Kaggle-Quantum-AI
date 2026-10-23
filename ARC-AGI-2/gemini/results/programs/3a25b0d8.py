# ARC-AGI-2 task 3a25b0d8: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

from collections import Counter, deque
import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    H, W = arr.shape

    # 1. Identify background color (most frequent)
    counts = Counter(arr.flatten().tolist())
    bg = counts.most_common(1)[0][0]

    # 2. Find 8-connected components of non-background cells
    visited = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if arr[r, c] != bg and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and arr[nr, nc] != bg and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                comps.append(cells)

    # Separate into wireframe component (single color) and colored component (multiple colors)
    mono_comps = [comp for comp in comps if len({arr[r, c] for r, c in comp}) == 1]
    poly_comps = [comp for comp in comps if len({arr[r, c] for r, c in comp}) > 1]

    wireframe_cells = max(mono_comps, key=len)
    border_color = int(arr[wireframe_cells[0][0], wireframe_cells[0][1]])
    colored_cells = [cell for comp in poly_comps for cell in comp]

    # 3. Extract bounding boxes for the wireframe and colored figures
    w_rs = [r for r, _ in wireframe_cells]
    w_cs = [c for _, c in wireframe_cells]
    w_rmin, w_rmax, w_cmin, w_cmax = min(w_rs), max(w_rs), min(w_cs), max(w_cs)
    w_h, w_w = w_rmax - w_rmin + 1, w_cmax - w_cmin + 1

    out = np.full((w_h, w_w), bg, dtype=int)
    for r, c in wireframe_cells:
        out[r - w_rmin, c - w_cmin] = border_color

    c_rs = [r for r, _ in colored_cells]
    c_cs = [c for _, c in colored_cells]
    c_rmin, c_rmax, c_cmin, c_cmax = min(c_rs), max(c_rs), min(c_cs), max(c_cs)
    c_h, c_w = c_rmax - c_rmin + 1, c_cmax - c_cmin + 1
    c_sub = arr[c_rmin:c_rmax + 1, c_cmin:c_cmax + 1]

    # 4. Find enclosed 4-connected background holes in the wireframe
    padded = np.full((w_h + 2, w_w + 2), bg, dtype=int)
    padded[1:-1, 1:-1] = out
    ext_vis = np.zeros((w_h + 2, w_w + 2), dtype=bool)
    q = deque([(0, 0)])
    ext_vis[0, 0] = True
    while q:
        cr, cc = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < w_h + 2 and 0 <= nc < w_w + 2 and not ext_vis[nr, nc] and padded[nr, nc] == bg:
                ext_vis[nr, nc] = True
                q.append((nr, nc))

    hole_vis = ext_vis[1:-1, 1:-1].copy()
    holes = []
    for r in range(w_h):
        for c in range(w_w):
            if out[r, c] == bg and not hole_vis[r, c]:
                q = deque([(r, c)])
                hole_vis[r, c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < w_h and 0 <= nc < w_w and out[nr, nc] == bg and not hole_vis[nr, nc]:
                            hole_vis[nr, nc] = True
                            q.append((nr, nc))
                holes.append(cells)

    # 5. Find 4-connected colored regions in the colored figure
    c_vis = np.zeros((c_h, c_w), dtype=bool)
    color_regions = []
    for r in range(c_h):
        for c in range(c_w):
            val = int(c_sub[r, c])
            if val != bg and val != border_color and not c_vis[r, c]:
                q = deque([(r, c)])
                c_vis[r, c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < c_h and 0 <= nc < c_w and not c_vis[nr, nc] and c_sub[nr, nc] == val:
                            c_vis[nr, nc] = True
                            q.append((nr, nc))
                mean_r = np.mean([pr for pr, _ in cells]) / max(1, c_h - 1)
                mean_c = np.mean([pc for _, pc in cells]) / max(1, c_w - 1)
                color_regions.append((mean_r, mean_c, val))

    # 6. Match each wireframe hole to the nearest colored region and fill
    for hole in holes:
        hr = np.mean([pr for pr, _ in hole]) / max(1, w_h - 1)
        hc = np.mean([pc for _, pc in hole]) / max(1, w_w - 1)
        best_color = min(
            color_regions,
            key=lambda reg: (reg[0] - hr) ** 2 + (reg[1] - hc) ** 2
        )[2]
        for r, c in hole:
            out[r, c] = best_color

    return out.tolist()
