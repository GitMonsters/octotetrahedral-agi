from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = Counter(grid[r][c] for r in range(H) for c in range(W)).most_common(1)[0][0]
    
    # Find shape cells
    shape = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                shape.append if False else shape.add((r, c))
    
    if not shape:
        return [row[:] for row in grid]
    
    min_r = min(r for r, c in shape)
    max_r = max(r for r, c in shape)
    min_c = min(c for r, c in shape)
    max_c = max(c for r, c in shape)
    
    # Try both axes and pick the one with fewer removals
    best_axis = None
    best_removals = float('inf')
    best_result = None
    
    # Try horizontal axis (reflect rows)
    for double_cr in range(min_r + max_r - 1, min_r + max_r + 2):
        # center_row = double_cr / 2
        removals = 0
        to_remove = set()
        for r, c in shape:
            mr = double_cr - r  # mirror row
            mc = c
            if (mr, mc) not in shape:
                removals += 1
                to_remove.add((r, c))
        if removals < best_removals:
            best_removals = removals
            best_result = to_remove
    
    # Try vertical axis (reflect cols)
    for double_cc in range(min_c + max_c - 1, min_c + max_c + 2):
        # center_col = double_cc / 2
        removals = 0
        to_remove = set()
        for r, c in shape:
            mr = r
            mc = double_cc - c  # mirror col
            if (mr, mc) not in shape:
                removals += 1
                to_remove.add((r, c))
        if removals < best_removals:
            best_removals = removals
            best_result = to_remove
    
    output = [row[:] for row in grid]
    if best_result:
        for r, c in best_result:
            output[r][c] = bg
    
    return output
