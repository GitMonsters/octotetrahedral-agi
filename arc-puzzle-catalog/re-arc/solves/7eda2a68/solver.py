from collections import Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    counts = Counter(c for r in grid for c in r)
    bg = counts.most_common(1)[0][0]
    out = [row[:] for row in grid]
    
    five_col_set = set()
    for c in range(cols):
        if all(grid[r][c] == 5 for r in range(rows)):
            five_col_set.add(c)
    five_cols = sorted(five_col_set)
    
    dots = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and c not in five_col_set:
                dots.append((r, c, grid[r][c]))
    
    for dr, dc, color in dots:
        left_five = right_five = None
        for fc in five_cols:
            if fc > dc:
                right_five = fc; break
        for fc in reversed(five_cols):
            if fc < dc:
                left_five = fc; break
        
        targets = []
        if left_five is not None: targets.append(left_five)
        if right_five is not None: targets.append(right_five)
        
        for fc in targets:
            for dr2 in range(dr-1, dr+2):
                for dc2 in range(fc-1, fc+2):
                    if 0 <= dr2 < rows and 0 <= dc2 < cols:
                        out[dr2][dc2] = 5
            out[dr][fc] = color
        
        if left_five is not None:
            for c in range(left_five + 2, dc):
                out[dr][c] = color
        if right_five is not None:
            for c in range(dc + 1, right_five - 1):
                out[dr][c] = color
    
    return out
