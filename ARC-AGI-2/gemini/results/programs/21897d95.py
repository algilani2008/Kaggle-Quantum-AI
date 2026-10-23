# ARC-AGI-2 task 21897d95: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

from collections import deque
import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    H, W = arr.shape
    cleaned = arr.copy()

    # 1. Detect arrow markers and restore underlying region colors
    markers = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(H):
        for c in range(W):
            for dr, dc in dirs:
                rt, ct = r + dr, c + dc
                rw1, cw1 = r - dc, c + dr
                rw2, cw2 = r + dc, c - dr
                rf1, cf1 = r + dr - dc, c + dc + dr
                rf2, cf2 = r + dr + dc, c + dc - dr

                coords = [(rt, ct), (rw1, cw1), (rw2, cw2), (rf1, cf1), (rf2, cf2)]
                if not all(0 <= nr < H and 0 <= nc < W for nr, nc in coords):
                    continue

                if (
                    arr[rt, ct] == 1
                    and arr[rw1, cw1] == 1
                    and arr[rw2, cw2] == 1
                    and arr[rf1, cf1] == arr[rf2, cf2]
                    and arr[rf1, cf1] != 1
                ):
                    bg = int(arr[rf1, cf1])
                    center_val = int(arr[r, c])
                    payload = bg if center_val == 1 else center_val
                    markers.append((r, c, dr, dc, bg, payload))
                    for nr, nc in [(r, c), (rt, ct), (rw1, cw1), (rw2, cw2)]:
                        cleaned[nr, nc] = bg

    # 2. Detect 3x3 L-shape rotation marker (if present)
    rot_k = 0  # np.rot90 k: 1 for 90 CCW, -1 for 90 CW
    visited_l = np.zeros((H, W), dtype=bool)
    for r in range(H - 2):
        for c in range(W - 2):
            sub = cleaned[r : r + 3, c : c + 3]
            # Top-right L-shape -> 90 deg CCW
            if (
                sub[0, 0] == sub[0, 1] == sub[0, 2] == sub[1, 2] == sub[2, 2]
                and sub[1, 0] == sub[1, 1] == sub[2, 0] == sub[2, 1]
                and sub[0, 0] != sub[1, 0]
            ):
                l_color = sub[0, 0]
                bg_color = sub[1, 0]
                # Check that the 5 pixels are an isolated component of l_color
                q = deque([(r, c)])
                comp = {(r, c)}
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in comp and cleaned[nr, nc] == l_color:
                            comp.add((nr, nc))
                            q.append((nr, nc))
                if len(comp) == 5:
                    rot_k = 1
                    for nr, nc in comp:
                        cleaned[nr, nc] = bg_color
                    break

            # Top-left L-shape -> 90 deg CW
            if (
                sub[0, 0] == sub[0, 1] == sub[0, 2] == sub[1, 0] == sub[2, 0]
                and sub[1, 1] == sub[1, 2] == sub[2, 1] == sub[2, 2]
                and sub[0, 0] != sub[1, 1]
            ):
                l_color = sub[0, 0]
                bg_color = sub[1, 1]
                q = deque([(r, c)])
                comp = {(r, c)}
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in comp and cleaned[nr, nc] == l_color:
                            comp.add((nr, nc))
                            q.append((nr, nc))
                if len(comp) == 5:
                    rot_k = -1
                    for nr, nc in comp:
                        cleaned[nr, nc] = bg_color
                    break

    # 3. Label 4-connected regions in the cleaned grid
    labels = np.full((H, W), -1, dtype=int)
    region_colors = []
    num_regions = 0
    for r in range(H):
        for c in range(W):
            if labels[r, c] == -1:
                color = int(cleaned[r, c])
                region_colors.append(color)
                q = deque([(r, c)])
                labels[r, c] = num_regions
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and labels[nr, nc] == -1 and cleaned[nr, nc] == color:
                            labels[nr, nc] = num_regions
                            q.append((nr, nc))
                num_regions += 1

    # 4. Apply color transfers from each marker to the target region it points at
    new_region_colors = list(region_colors)
    for r, c, dr, dc, bg, payload in markers:
        src_reg = labels[r, c]
        nr, nc = r + 2 * dr, c + 2 * dc
        while 0 <= nr < H and 0 <= nc < W and labels[nr, nc] == src_reg:
            nr += dr
            nc += dc
        if 0 <= nr < H and 0 <= nc < W:
            target_reg = labels[nr, nc]
            new_region_colors[target_reg] = payload

    # 5. Build output grid and apply rotation if needed
    out = np.zeros_like(cleaned)
    for reg_id, color in enumerate(new_region_colors):
        out[labels == reg_id] = color

    if rot_k != 0:
        out = np.rot90(out, rot_k)

    return out.tolist()
