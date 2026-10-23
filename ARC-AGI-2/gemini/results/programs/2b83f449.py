# ARC-AGI-2 task 2b83f449: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])
    out = [row[:] for row in grid]

    starts = []
    for r in range(0, H, 2):
        for c in range(W):
            if out[r][c] == 3:
                starts.append((r, c))
                out[r][c] = 8

    down_bridges = set()
    for r in range(1, H, 2):
        for c in range(1, W - 1):
            if grid[r][c - 1] == 7 and grid[r][c] == 7 and grid[r][c + 1] == 7:
                out[r][c - 1] = 8
                out[r][c] = 6
                out[r][c + 1] = 8
                out[r - 1][c] = 6
                out[r + 1][c] = 6
                down_bridges.add((r - 1, c))

    for r_start, c_start in starts:
        direction = 1 if c_start == 0 else -1
        curr_r, curr_c, curr_dir = r_start, c_start, direction
        moved_down = False

        while True:
            next_c = curr_c + curr_dir
            while 0 <= next_c < W and out[curr_r][next_c] not in (0, 6):
                next_c += curr_dir

            if 0 <= next_c < W and (curr_r, next_c) in down_bridges:
                curr_r += 2
                curr_c = next_c
                curr_dir = -curr_dir
                moved_down = True
            else:
                if not moved_down:
                     place_c = c_start
                     step = direction
                else:
                    place_c = next_c - curr_dir
                    step = -curr_dir

                while out[curr_r][place_c] == 3:
                    place_c += step
                out[curr_r][place_c] = 3
                break

    return out
