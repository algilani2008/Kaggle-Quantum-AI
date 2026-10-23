# ARC-AGI-2 task 2d0172a1: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

from collections import Counter, deque
import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    H, W = arr.shape

    counts = Counter(arr.flatten())
    bg = counts.most_common(1)[0][0]
    fg = [c for c in counts if c != bg][0]

    # Find 4-connected components of foreground pixels
    visited = np.zeros((H, W), dtype=bool)
    components = []
    for r in range(H):
        for c in range(W):
            if arr[r, c] == fg and not visited[r, c]:
                comp = []
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and arr[nr, nc] == fg and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                components.append(comp)

    # Determine the filled interior of each component to separate loops from blobs
    loops = []
    blobs = []
    for comp in components:
        padded = np.zeros((H + 2, W + 2), dtype=bool)
        for r, c in comp:
            padded[r + 1, c + 1] = True

        ext = np.zeros((H + 2, W + 2), dtype=bool)
        q = deque([(0, 0)])
        ext[0, 0] = True
        while q:
            cr, cc = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H + 2 and 0 <= nc < W + 2 and not padded[nr, nc] and not ext[nr, nc]:
                    ext[nr, nc] = True
                    q.append((nr, nc))

        filled = set()
        for r in range(H):
            for c in range(W):
                if not ext[r + 1, c + 1]:
                    filled.add((r, c))

        if len(filled) > len(comp):
            loops.append({"comp": set(comp), "filled": filled})
        else:
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            blobs.append({
                "comp": set(comp),
                "r_span": (min(rs), max(rs)),
                "c_span": (min(cs), max(cs)),
                "rep": comp[0],
            })

    # Cluster intervals by overlap to assign logical rows and columns to blobs
    def cluster_intervals(intervals):
        sorted_ivs = sorted(intervals, key=lambda x: (x[0], x[1]))
        merged = []
        for iv in sorted_ivs:
            if not merged or iv[0] > merged[-1][1]:
                merged.append([iv[0], iv[1]])
            else:
                merged[-1][1] = max(merged[-1][1], iv[1])
        return merged

    row_clusters = cluster_intervals([b["r_span"] for b in blobs])
    col_clusters = cluster_intervals([b["c_span"] for b in blobs])

    for b in blobs:
        r0, r1 = b["r_span"]
        c0, c1 = b["c_span"]
        b["l_row"] = next(i for i, ( mr0, mr1) in enumerate(row_clusters) if max(r0, mr0) <= min(r1, mr1))
        b["l_col"] = next(i for i, (mc0, mc1) in enumerate(col_clusters) if max(c0, mc0) <= min(c1, mc1))

    # Determine logical bounding boxes for each loop
    for idx, loop in enumerate(loops):
        loop["id"] = idx
        loop["area"] = len(loop["filled"])
        enclosed = [b for b in blobs if b["rep"] in loop["filled"]]
        loop["r_min"] = min(b["l_row"] for b in enclosed)
        loop["r_max"] = max(b["l_row"] for b in enclosed)
        loop["c_min"] = min(b["l_col"] for b in enclosed)
        loop["c_max"] = max(b["l_col"] for b in enclosed)

    # Build 1D coordinate mappings for rows and columns
    def build_axis(num_logical, min_key, max_key):
        events = []
        for i in range(num_logical):
            starts = [L for L in loops if L[min_key] == i]
            starts.sort(key=lambda L: -L["area"])
            for L in starts:
                events.append(("start", L["id"]))

            events.append(("point", i))

            ends = [L for L in loops if L[max_key] == i]
            ends.sort(key=lambda L: L["area"])
            for L in ends:
                events.append(("end", L["id"]))

        offset = 1 if events[0][0] == "point" else 0
        start_pos = {}
        end_pos = {}
        point_pos = {}
        for idx, (etype, eid) in enumerate(events):
            pos = offset + 2 * idx
            if etype == "start":
                start_pos[eid] = pos
            elif etype == "end":
                end_pos[eid] = pos
            else:
                point_pos[eid] = pos

        total_len = offset + 2 * len(events) - 1 + (1 if events[-1][0] == "point" else 0)
        return total_len, start_pos, end_pos, point_pos

    out_H, r_start, r_end, r_point = build_axis(len(row_clusters), "r_min", "r_max")
    out_W, c_start, c_end, c_point = build_axis(len(col_clusters), "c_min", "c_max")

    # Render the canonical output grid
    out = np.full((out_H, out_W), bg, dtype=int)
    for L in loops:
        lid = L["id"]
        r0, r1 = r_start[lid], r_end[lid]
        c0, c1 = c_start[lid], c_end[lid]
        out[r0, c0 : c1 + 1] = fg
        out[r1, c0 : c1 + 1] = fg
        out[r0 : r1 + 1, c0] = fg
        out[r0 : r1 + 1, c1] = fg

    for b in blobs:
        out[r_point[b["l_row"]], c_point[b["l_col"]]] = fg

    return out.tolist()
