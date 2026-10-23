# ARC-AGI-2 task 36a08778: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

def transform(grid: list[list[int]]) -> list[list[int]]:
    res = [row[:] for row in grid]
    h = len(res)
    w = len(res[0])

    for r in range(h - 1):
        # Find current stream positions in row r that are blocked from below
        sources = [c for c in range(w) if res[r][c] == 6 and res[r + 1][c] == 2]
        for c in sources:
            # Spread left
            for c_left in range(c - 1, -1, -1):
                if res[r][c_left] == 2:
                    break
                res[r][c_left] = 6
                if res[r + 1][c_left] != 2:
                    break
            # Spread right
            for c_right in range(c + 1, w):
                if res[r][c_right] == 2:
                    break
                res[r][c_right] = 6
                if res[r + 1][c_right] != 2:
                    break

        # Flow downward into row r + 1 where unblocked
        for c in range(w):
            if res[r][c] == 6 and res[r + 1][c] == 7:
                res[r + 1][c] = 6

    return res
