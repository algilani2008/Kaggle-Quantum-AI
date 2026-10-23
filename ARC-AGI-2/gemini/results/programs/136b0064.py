# ARC-AGI-2 task 136b0064: verified on all train pairs
# model=gemini-4-argon sample=1 round=0 test_correct=1.0

def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    sep_col = next(c for c in range(len(grid[0])) if all(grid[r][c] == 4 for r in range(H)))
    
    out = [row[sep_col + 1:] for row in grid]
    c = out[0].index(5)
    r = 1
    
    for bc in range(0, sep_col, 4):
        for br in range(0, H, 4):
            block = [grid[br + dr][bc : bc + 3] for dr in range(3)]
            color = next((val for row in block for val in row if val != 0), 0)
            
            if color == 1:
                for dc in range(3):
                    out[r][c + dc] = 1
                c += 2
                r += 1
            elif color == 2:
                for dc in range(2):
                    out[r][c - dc] = 2
                c -= 1
                r += 1
            elif color == 3:
                for dc in range(4):
                    out[r][c - dc] = 3
                c -= 3
                r += 1
            elif color == 6:
                out[r][c] = 6
                out[r + 1][c] = 6
                r += 2
                
    return out
