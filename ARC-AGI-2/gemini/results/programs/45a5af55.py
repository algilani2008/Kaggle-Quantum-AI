# ARC-AGI-2 task 45a5af55: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

def transform(grid: list[list[int]]) -> list[list[int]]:
    h = len(grid)
    n = 2 * h - 2
    return [
        [grid[min(r, c, n - 1 - r, n - 1 - c)][0] for c in range(n)]
        for r in range(n)
    ]
