# ARC-AGI-2 task 271d71e2: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

import numpy as np


def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    out = arr.copy()
    H, W = arr.shape

    visited = np.zeros((H, W), dtype=bool)
    boxes = []

    for r in range(H):
        for c in range(W):
            if arr[r, c] == 0 and not visited[r, c]:
                # Find bounding box of this 0-bordered rectangle
                r2 = r
                while r2 + 1 < H and arr[r2 + 1, c] == 0:
                    r2 += 1
                c2 = c
                while c2 + 1 < W and arr[r, c2 + 1] == 0:
                    c2 += 1
                visited[r : r2 + 1, c : c2 + 1] = True
                boxes.append((r, r2, c, c2))

    for r1, r2, c1, c2 in boxes:
        interior = arr[r1 + 1 : r2, c1 + 1 : c2]
        num_5 = int(np.sum(interior == 5))
        num_7 = int(np.sum(interior == 7))

        # Determine movement direction and distance between the two 9-bars
        if r1 > 0 and np.all(arr[r1 - 1, c1 : c2 + 1] == 9):
            direction = "UP"
            r_adj = r1 - 1
            r_targ = r_adj - 1
            while not np.all(arr[r_targ, c1 : c2 + 1] == 9):
                r_targ -= 1
            dist = r_adj - r_targ
            dr, dc = -1, 0
            rot_k = 0
        elif r2 + 1 < H and np.all(arr[r2 + 1, c1 : c2 + 1] == 9):
            direction = "DOWN"
            r_adj = r2 + 1
            r_targ = r_adj + 1
            while not np.all(arr[r_targ, c1 : c2 + 1] == 9):
                r_targ += 1
            dist = r_targ - r_adj
            dr, dc = 1, 0
            rot_k = 2
        elif c1 > 0 and np.all(arr[r1 : r2 + 1, c1 - 1] == 9):
            direction = "LEFT"
            c_adj = c1 - 1
            c_targ = c_adj - 1
            while not np.all(arr[r1 : r2 + 1, c_targ] == 9):
                c_targ -= 1
            dist = c_adj - c_targ
            dr, dc = 0, -1
            rot_k = 3
        elif c2 + 1 < W and np.all(arr[r1 : r2 + 1, c2 + 1] == 9):
            direction = "RIGHT"
            c_adj = c2 + 1
            c_targ = c_adj + 1
            while not np.all(arr[r1 : r2 + 1, c_targ] == 9):
                c_targ += 1
            dist = c_targ - c_adj
            dr, dc = 0, 1
            rot_k = 1
        else:
            continue

        steps = min(dist, num_5)
        total_7 = num_7 + steps

        # Construct the updated interior in canonical orientation (facing UP)
        rot_int = np.rot90(interior, k=rot_k)
        flat = np.full(rot_int.size, 5, dtype=int)
        flat[:total_7] = 7
        new_rot_int = flat.reshape(rot_int.shape)
        new_interior = np.rot90(new_rot_int, k=-rot_k)

        # Clear original box and its adjacent 9-bar
        out[r1 : r2 + 1, c1 : c2 + 1] = 6
        if direction == "UP":
            out[r1 - 1, c1 : c2 + 1] = 6
        elif direction == "DOWN":
            out[r2 + 1, c1 : c2 + 1] = 6
        elif direction == "LEFT":
            out[r1 : r2 + 1, c1 - 1] = 6
        elif direction == "RIGHT":
            out[r1 : r2 + 1, c2 + 1] = 6

        # Draw moved box and moved 9-bar
        nr1, nr2 = r1 + steps * dr, r2 + steps * dr
        nc1, nc2 = c1 + steps * dc, c2 + steps * dc
        out[nr1 : nr2 + 1, nc1 : nc2 + 1] = 0
        out[nr1 + 1 : nr2, nc1 + 1 : nc2] = new_interior

        if direction == "UP":
            out[nr1 - 1, nc1 : nc2 + 1] = 9
        elif direction == "DOWN":
            out[nr2 + 1, nc1 : nc2 + 1] = 9
        elif direction == "LEFT":
            out[nr1 : nr2 + 1, nc1 - 1] = 9
        elif direction == "RIGHT":
            out[nr1 : nr2 + 1, nc2 + 1] = 9

    return out.tolist()
