# ARC-AGI-2 task 4a21e3da: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid)
    H, W = arr.shape
    out = np.ones((H, W), dtype=int)
    
    # Locate laser sources on the four borders
    top_c = None
    bottom_c = None
    left_r = None
    right_r = None
    
    for c in range(W):
        if arr[0, c] == 2:
            top_c = c
        if arr[H - 1, c] == 2:
            bottom_c = c
    for r in range(H):
        if arr[r, 0] == 2:
            left_r = r
        if arr[r, W - 1] == 2:
            right_r = r
            
    v_cut = top_c if top_c is not None else bottom_c
    h_cut = left_r if left_r is not None else right_r
    
    # Draw vertical laser beam
    if top_c is not None:
        c = top_c
        sevens = np.where(arr[:, c] == 7)[0]
        max_r = sevens.max() if len(sevens) > 0 else 0
        for r in range(0, max_r + 1):
            out[r, c] = 7 if arr[r, c] == 7 else 2
            
    if bottom_c is not None:
        c = bottom_c
        sevens = np.where(arr[:, c] == 7)[0]
        min_r = sevens.min() if len(sevens) > 0 else H - 1
        for r in range(min_r, H):
            out[r, c] = 7 if arr[r, c] == 7 else 2
            
    # Draw horizontal laser beam
    if left_r is not None:
        r = left_r
        sevens = np.where(arr[r, :] == 7)[0]
        max_c = sevens.max() if len(sevens) > 0 else 0
        for c in range(0, max_c + 1):
            out[r, c] = 7 if arr[r, c] == 7 else 2
            
    if right_r is not None:
        r = right_r
        sevens = np.where(arr[r, :] == 7)[0]
        min_c = sevens.min() if len(sevens) > 0 else W - 1
        for c in range(min_c, W):
            out[r, c] = 7 if arr[r, c] == 7 else 2

    # Helper to shift a piece's bounding box into a corner
    def place_in_corner(mask, align_bottom, align_right):
        rs, cs = np.where(mask & (arr == 7))
        if len(rs) == 0:
            return
        r_shift = (H - 1 - rs.max()) if align_bottom else -rs.min()
        c_shift = (W - 1 - cs.max()) if align_right else -cs.min()
        for r, c in zip(rs, cs):
            out[r + r_shift, c + c_shift] = 7

    R_idx, C_idx = np.indices((H, W))

    if v_cut is not None and h_cut is not None:
        if top_c is not None or left_r is not None:
            place_in_corner((R_idx < h_cut) & (C_idx < v_cut), align_bottom=False, align_right=False)
        if top_c is not None or right_r is not None:
            place_in_corner((R_idx < h_cut) & (C_idx > v_cut), align_bottom=False, align_right=True)
        if bottom_c is not None or left_r is not None:
            place_in_corner((R_idx > h_cut) & (C_idx < v_cut), align_bottom=True, align_right=False)
        if bottom_c is not None or right_r is not None:
            place_in_corner((R_idx > h_cut) & (C_idx > v_cut), align_bottom=True, align_right=True)
    elif v_cut is not None:
        align_bottom = (bottom_c is not None)
        place_in_corner(C_idx < v_cut, align_bottom=align_bottom, align_right=False)
        place_in_corner(C_idx > v_cut, align_bottom=align_bottom, align_right=True)
    elif h_cut is not None:
        align_right = (right_r is not None)
        place_in_corner(R_idx < h_cut, align_bottom=False, align_right=align_right)
        place_in_corner(R_idx > h_cut, align_bottom=True, align_right=align_right)

    return out.tolist()
