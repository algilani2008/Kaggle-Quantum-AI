# ARC-AGI-2 task 38007db0: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

from collections import Counter


def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])
    sep_color = grid[0][0]

    sep_rows = [r for r in range(H) if all(grid[r][c] == sep_color for c in range(W))]
    sep_cols = [c for c in range(W) if all(grid[r][c] == sep_color for r in range(H))]

    sub_w = sep_cols[1] - sep_cols[0] - 1
    out_w = sub_w + 2
    out = [[sep_color] * out_w]

    for i in range(len(sep_rows) - 1):
        r1, r2 = sep_rows[i] + 1, sep_rows[i + 1]
        subgrids = []
        for j in range(len(sep_cols) - 1):
            c1, c2 = sep_cols[j] + 1, sep_cols[j + 1]
            sub = tuple(tuple(grid[r][c1:c2]) for r in range(r1, r2))
            subgrids.append(sub)

        counts = Counter(subgrids)
        unique_sub = next(sub for sub, cnt in counts.items() if cnt == 1)

        for row in unique_sub:
            out.append([sep_color] + list(row) + [sep_color])
        out.append([sep_color] * out_w)

    return out
