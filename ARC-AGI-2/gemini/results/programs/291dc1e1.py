# ARC-AGI-2 task 291dc1e1: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    H, W = arr.shape

    # Locate the '0' corner on the border
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    r0, c0 = next((r, c) for r, c in corners if arr[r, c] == 0)

    # Determine whether the '1'-axis is horizontal (along row r0) or vertical (along col c0)
    adj_r = 1 if r0 == 0 else H - 2
    adj_c = 1 if c0 == 0 else W - 2
    one_is_horizontal = (arr[r0, adj_c] == 1)

    # Extract interior slices (excluding the border row r0 and border col c0)
    rows = [r for r in range(H) if r != r0]
    cols = [c for c in range(W) if c != c0]

    objects = []

    if one_is_horizontal:
        # Tracks are horizontal bands of rows (indexed along the vertical '2'-axis)
        r_idx = 0
        tracks = []
        while r_idx < len(rows):
            r = rows[r_idx]
            if np.any(arr[r, cols] != 8):
                track_rows = [r]
                r_idx += 1
                while r_idx < len(rows) and np.any(arr[rows[r_idx], cols] != 8):
                    track_rows.append(rows[r_idx])
                    r_idx += 1
                tracks.append(track_rows)
            else:
                r_idx += 1

        if r0 == H - 1:
            tracks.reverse()

        for track_rows in tracks:
            c_idx = 0
            track_objs = []
            while c_idx < len(cols):
                c = cols[c_idx]
                if np.any(arr[track_rows, c] != 8):
                    obj_cols = [c]
                    c_idx += 1
                    while c_idx < len(cols) and np.any(arr[track_rows, cols[c_idx]] != 8):
                        obj_cols.append(cols[c_idx])
                        c_idx += 1
                    obj = arr[np.ix_(track_rows, obj_cols)]
                    track_objs.append(obj)
                else:
                    c_idx += 1
            if c0 == W - 1:
                track_objs.reverse()
            objects.extend(track_objs)
    else:
        # Tracks are vertical bands of columns (indexed along the horizontal '2'-axis)
        c_idx = 0
        tracks = []
        while c_idx < len(cols):
            c = cols[c_idx]
            if np.any(arr[rows, c] != 8):
                track_cols = [c]
                c_idx += 1
                while c_idx < len(cols) and np.any(arr[rows, cols[c_idx]] != 8):
                    track_cols.append(cols[c_idx])
                    c_idx += 1
                tracks.append(track_cols)
            else:
                c_idx += 1

        if c0 == W - 1:
            tracks.reverse()

        for track_cols in tracks:
            r_idx = 0
            track_objs = []
            while r_idx < len(rows):
                r = rows[r_idx]
                if np.any(arr[r, track_cols] != 8):
                    obj_rows = [r]
                    r_idx += 1
                    while r_idx < len(rows) and np.any(arr[rows[r_idx], track_cols] != 8):
                        obj_rows.append(rows[r_idx])
                        r_idx += 1
                    obj = arr[np.ix_(obj_rows, track_cols)]
                    # Rotate 90 degrees counter-clockwise to make the object horizontal
                    track_objs.append(np.rot90(obj, 1))
                else:
                    r_idx += 1
            if r0 == H - 1:
                track_objs.reverse()
            objects.extend(track_objs)

    max_w = max(obj.shape[1] for obj in objects)
    out_strips = []
    for obj in objects:
        h, w = obj.shape
        strip = np.full((h, max_w), 8, dtype=int)
        pad = (max_w - w) // 2
        strip[:, pad:pad + w] = obj
        out_strips.append(strip)

    return np.vstack(out_strips).tolist()
