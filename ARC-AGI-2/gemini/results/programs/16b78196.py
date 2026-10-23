# ARC-AGI-2 task 16b78196: verified on all train pairs
# model=gemini-4-argon sample=1 round=1 test_correct=1.0

import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    H, W = arr.shape

    # Find monochromatic 8-connected components
    visited = np.zeros((H, W), dtype=bool)
    components = []
    for r in range(H):
        for c in range(W):
            if arr[r, c] != 0 and not visited[r, c]:
                color = arr[r, c]
                comp = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop()
                    comp.append((cr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                if not visited[nr, nc] and arr[nr, nc] == color:
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))
                components.append((color, comp))

    # Identify the wall component (spans across the grid horizontally or vertically)
    wall_idx = -1
    is_vertical = False
    for i, (color, comp) in enumerate(components):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        if min(cs) == 0 and max(cs) == W - 1:
            wall_idx = i
            is_vertical = False
            break
        if min(rs) == 0 and max(rs) == H - 1:
            wall_idx = i
            is_vertical = True
            break

    if is_vertical:
        return np.array(transform(arr.T.tolist())).T.tolist()

    # Build base grid with only the horizontal wall
    base = np.zeros((H, W), dtype=int)
    wall_color, wall_comp = components[wall_idx]
    for r, c in wall_comp:
        base[r, c] = wall_color

    # Extract movable pieces as cropped 2D arrays
    pieces = []
    for i, (color, comp) in enumerate(components):
        if i == wall_idx:
            continue
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in comp:
            patch[r - r0, c - c0] = color
        pieces.append(patch)

    num_pieces = len(pieces)
    valid_stacks = []

    # Search for valid stacks on both 'bottom' and 'top' (by vertically flipping 'top')
    for side in ("bottom", "top"):
        work_base = base if side == "bottom" else base[::-1, :]
        work_pieces = pieces if side == "bottom" else [p[::-1, :] for p in pieces]

        widths = sorted({p.shape[1] for p in work_pieces})
        for w in widths:
            for c in range(W - w + 1):
                strip = work_base[:, c : c + w]
                full_rows = np.where(np.all(strip != 0, axis=1))[0]
                if len(full_rows) == 0:
                    continue
                r_solid = int(full_rows[-1])
                non_zero_rows = np.where(np.any(strip != 0, axis=1))[0]
                r_last_wall = int(non_zero_rows[-1])

                # Must be an actual indentation below r_solid
                if r_last_wall <= r_solid:
                    continue

                def dfs(cur_strip, r_gap, r_max, used_mask, placements):
                    for idx, p in enumerate(work_pieces):
                        if (used_mask >> idx) & 1:
                            continue
                        ph, pw = p.shape
                        if pw != w or r_gap + ph > H:
                            continue
                        target = cur_strip[r_gap : r_gap + ph, :]
                        # No overlap allowed
                        if np.any((target != 0) & (p != 0)):
                            continue
                        # Row r_gap must become completely full
                        if np.any((target[0, :] == 0) & (p[0, :] == 0)):
                            continue

                        next_strip = cur_strip.copy()
                        next_strip[r_gap : r_gap + ph, :] += p
                        next_max = max(r_max, r_gap + ph - 1)
                        next_used = used_mask | (1 << idx)
                        next_placements = placements + [(side, c, r_gap, idx)]

                        # Find next row with a 0 between r_gap and next_max
                        sub = next_strip[r_gap : next_max + 1, :]
                        zero_rows = np.where(np.any(sub == 0, axis=1))[0]
                        if len(zero_rows) == 0:
                            valid_stacks.append((next_used, next_placements))
                        else:
                            dfs(
                                next_strip,
                                r_gap + int(zero_rows[0]),
                                next_max,
                                next_used,
                                next_placements,
                            )

                dfs(strip.copy(), r_solid + 1, r_last_wall, 0, [])

    # Partition all pieces into disjoint valid stacks
    full_mask = (1 << num_pieces) - 1
    chosen_placements = []

    def solve_partition(idx, mask, acc):
        if mask == full_mask:
            chosen_placements.extend(acc)
            return True
        for i in range(idx, len(valid_stacks)):
            smask, splacements = valid_stacks[i]
            if (mask & smask) == 0:
                if solve_partition(i + 1, mask | smask, acc + splacements):
                    return True
        return False

    solve_partition(0, 0, [])

    # Render the chosen placements onto the base grid
    out = base.copy()
    for side, c, r_gap, idx in chosen_placements:
        p = pieces[idx]
        ph, pw = p.shape
        r_top = r_gap if side == "bottom" else H - r_gap - ph
        mask = p != 0
        out[r_top : r_top + ph, c : c + pw][mask] = p[mask]

    return out.tolist()
