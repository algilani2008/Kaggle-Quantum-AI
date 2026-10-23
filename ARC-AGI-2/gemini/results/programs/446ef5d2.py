# ARC-AGI-2 task 446ef5d2: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

from collections import Counter, deque
import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    R, C = arr.shape

    # 1. Background colour is the most frequent colour
    counts = Counter(arr.flatten())
    bg = counts.most_common(1)[0][0]

    # 2. Find connected components of identical colour to locate the 3-cell L-marker
    visited = np.zeros((R, C), dtype=bool)
    marker_cells = None
    corner_cell = None
    marker_box = None

    for r in range(R - 1):
        for c in range(C - 1):
            window = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            vals = [arr[pr, pc] for pr, pc in window]
            if bg in vals:
                continue
            val_counts = Counter(vals)
            if len(val_counts) == 2 and 3 in val_counts.values():
                m_col = [k for k, v in val_counts.items() if v == 3][0]
                if counts[m_col] == 3:
                    marker_cells = [(pr, pc) for pr, pc in window if arr[pr, pc] == m_col]
                    corner_cell = [(pr, pc) for pr, pc in window if arr[pr, pc] != m_col][0]
                    marker_box = (r, c)
                    break
        if marker_cells is not None:
            break

    # Remove marker cells and extract all 4-connected non-background pieces
    work = arr.copy()
    for mr, mc in marker_cells:
        work[mr, mc] = bg

    pieces = []
    fixed_idx = None
    vis = np.zeros((R, C), dtype=bool)
    for r in range(R):
        for c in range(C):
            if work[r, c] != bg and not vis[r, c]:
                q = deque([(r, c)])
                vis[r, c] = True
                comp = []
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc, int(work[cr, cc])))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < R and 0 <= nc < C and not vis[nr, nc] and work[nr, nc] != bg:
                            vis[nr, nc] = True
                            q.append((nr, nc))
                comp.sort()
                if any((cr, cc) == corner_cell for cr, cc, _ in comp):
                    fixed_idx = len(pieces)
                pieces.append(comp)

    total_area = sum(len(p) for p in pieces)
    r_c, c_c = corner_cell
    r_m, c_m = marker_box
    border_color = int(work[r_c, c_c])

    # Normalize each piece relative to its first cell in row-major order
    norm_pieces = []
    for p in pieces:
        r0, c0, _ = p[0]
        rel = [(r - r0, c - c0, val) for r, c, val in p]
        rs = [r for r, _, _ in p]
        cs = [c for _, c, _ in p]
        h = max(rs) - min(rs) + 1
        w = max(cs) - min(cs) + 1
        norm_pieces.append((rel, h, w, p))

    max_h = max(h for _, h, _, _ in norm_pieces)
    max_w = max(w for _, _, w, _ in norm_pieces)

    best_score = -1
    best_board = None
    best_rect_origin = None

    for H in range(max_h, R + 1):
        if total_area % H != 0:
            continue
        W = total_area // H
        if W < max_w or W > C:
            continue

        # Determine top-left of the target HxW rectangle in the grid
        r_top = r_c - H + 1 if r_c == r_m else r_c
        c_left = c_c - W + 1 if c_c == c_m else c_c
        if not (0 <= r_top and r_top + H <= R and 0 <= c_left and c_left + W <= C):
            continue

        board = np.full((H, W), -1, dtype=int)
        owner = np.full((H, W), -1, dtype=int)

        # Place the anchored piece at its original grid coordinates
        possible = True
        for pr, pc, val in pieces[fixed_idx]:
            br, bc = pr - r_top, pc - c_left
            if not (0 <= br < H and 0 <= bc < W):
                possible = False
                break
            board[br, bc] = val
            owner[br, bc] = fixed_idx
        if not possible:
            continue

        rem_indices = [i for i in range(len(pieces)) if i != fixed_idx]
        used = [False] * len(rem_indices)

        def solve(first_idx: int):
            nonlocal best_score, best_board, best_rect_origin
            while first_idx < H * W and board[first_idx // W, first_idx % W] != -1:
                first_idx += 1

            if first_idx == H * W:
                # Score the completed tiling by internal seam matches + uniform border matches
                score = 0
                for r in range(H):
                    for c in range(W):
                        if r in (0, H - 1) or c in (0, W - 1):
                            if board[r, c] == border_color:
                                score += 1
                        if r + 1 < H and owner[r, c] != owner[r + 1, c]:
                            if board[r, c] == board[r + 1, c]:
                                score += 2
                        if c + 1 < W and owner[r, c] != owner[r, c + 1]:
                            if board[r, c] == board[r, c + 1]:
                                score += 2
                if score > best_score:
                    best_score = score
                    best_board = board.copy()
                    best_rect_origin = (r_top, c_left)
                return

            r0, c0 = divmod(first_idx, W)
            for k, p_idx in enumerate(rem_indices):
                if used[k]:
                    continue
                rel, _, _, _ = norm_pieces[p_idx]
                fits = True
                for dr, dc, _ in rel:
                    nr, nc = r0 + dr, c0 + dc
                    if not (0 <= nr < H and 0 <= nc < W and board[nr, nc] == -1):
                        fits = False
                        break
                if not fits:
                    continue

                used[k] = True
                for dr, dc, val in rel:
                    board[r0 + dr, c0 + dc] = val
                    owner[r0 + dr, c0 + dc] = p_idx

                solve(first_idx + 1)

                for dr, dc, _ in rel:
                    board[r0 + dr, c0 + dc] = -1
                    owner[r0 + dr, c0 + dc] = -1
                used[k] = False

        solve(0)

    out = np.full((R, C), bg, dtype=int)
    r_top, c_left = best_rect_origin
    H, W = best_board.shape
    out[r_top : r_top + H, c_left : c_left + W] = best_board
    return out.tolist()
