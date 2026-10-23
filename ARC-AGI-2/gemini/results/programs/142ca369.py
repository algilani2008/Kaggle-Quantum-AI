# ARC-AGI-2 task 142ca369: verified on all train pairs
# model=gemini-4-argon sample=1 round=2 test_correct=1.0

import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    H, W = arr.shape
    output = arr.copy()

    visited = np.zeros((H, W), dtype=bool)
    emitters = []
    reflectors = {}

    # Find monochromatic 4-connected components
    for r in range(H):
        for c in range(W):
            if arr[r, c] != 0 and not visited[r, c]:
                color = int(arr[r, c])
                comp = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and arr[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                rows = [p[0] for p in comp]
                cols = [p[1] for p in comp]
                # L-shape emitter: 3 cells in a 2x2 bounding box
                if len(comp) == 3 and (max(rows) - min(rows) == 1) and (max(cols) - min(cols) == 1):
                    comp_set = set(comp)
                    for pr, pc in comp:
                        neighbors = [
                            (nr, nc)
                            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
                            if (nr, nc) in comp_set
                        ]
                        if len(neighbors) == 2:
                            dr = 2 * pr - neighbors[0][0] - neighbors[1][0]
                            dc = 2 * pc - neighbors[0][1] - neighbors[1][1]
                            emitters.append((pr, pc, dr, dc, color))
                            break
                else:
                    # Reflector (1x1 dot, 1x3 bar, or 3x1 bar): record its center cell
                    center_r = sum(rows) // len(comp)
                    center_c = sum(cols) // len(comp)
                    reflectors[(center_r, center_c)] = color

    # Trace rays from each emitter
    for vr, vc, dr, dc, color in emitters:
        r, c = vr + dr, vc + dc
        cur_color = color
        while 0 <= r < H and 0 <= c < W:
            if (r + dr, c) in reflectors:
                cur_color = reflectors[(r + dr, c)]
                dr = -dr
            elif (r, c + dc) in reflectors:
                cur_color = reflectors[(r, c + dc)]
                dc = -dc
            output[r, c] = cur_color
            r += dr
            c += dc

    return output.tolist()
