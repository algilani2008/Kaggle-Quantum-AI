# ARC-AGI-2 task 28a6681f: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])

    # Keep all static obstacles and remove water (1s) to simulate settling
    out = [[0 if cell == 1 else cell for cell in row] for row in grid]

    # Process water units from bottom to top
    water_positions = [
        (r, c)
        for r in range(H - 1, -1, -1)
        for c in range(W)
        if grid[r][c] == 1
    ]

    for start_r, start_c in water_positions:
        r, c = start_r, start_c
        while True:
            # Fall straight down while possible
            while r + 1 < H and out[r + 1][c] == 0:
                r += 1

            # Scan left and right for a drop-off
            left_drop = None
            left_bound = c
            for cl in range(c - 1, -1, -1):
                if out[r][cl] != 0:
                    break
                left_bound = cl
                if r + 1 < H and out[r + 1][cl] == 0:
                    left_drop = cl
                    break

            right_drop = None
            right_bound = c
            for cr in range(c + 1, W):
                if out[r][cr] != 0:
                    break
                right_bound = cr
                if r + 1 < H and out[r + 1][cr] == 0:
                    right_drop = cr
                    break

            if left_drop is not None and right_drop is not None:
                c = left_drop if (c - left_drop <= right_drop - c) else right_drop
            elif left_drop is not None:
                c = left_drop
            elif right_drop is not None:
                c = right_drop
            else:
                # Settle inside the enclosed horizontal basin at row r
                candidates = [
                    col for col in range(left_bound, right_bound + 1)
                    if out[r][col] == 0
                ]
                target_c = max(candidates, key=lambda col: abs(col - c))
                out[r][target_c] = 1
                break

    return out
