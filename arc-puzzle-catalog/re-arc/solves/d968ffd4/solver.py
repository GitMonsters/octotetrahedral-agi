"""ARC puzzle d968ffd4 solver"""
from collections import Counter

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    grid = [row[:] for row in input_grid]
    rows, cols = len(grid), len(grid[0])
    
    # Background = most common color
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    
    # Find bounding boxes of the two colored objects
    bboxes: dict[int, dict] = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != bg:
                if v not in bboxes:
                    bboxes[v] = {'min_r': r, 'max_r': r, 'min_c': c, 'max_c': c}
                else:
                    bb = bboxes[v]
                    bb['min_r'] = min(bb['min_r'], r)
                    bb['max_r'] = max(bb['max_r'], r)
                    bb['min_c'] = min(bb['min_c'], c)
                    bb['max_c'] = max(bb['max_c'], c)
    
    colors = list(bboxes.keys())
    c1, c2 = colors[0], colors[1]
    bb1, bb2 = bboxes[c1], bboxes[c2]
    
    if bb1['max_c'] < bb2['min_c'] or bb2['max_c'] < bb1['min_c']:
        # Horizontally separated
        if bb1['min_c'] < bb2['min_c']:
            lc, rc, lbb, rbb = c1, c2, bb1, bb2
        else:
            lc, rc, lbb, rbb = c2, c1, bb2, bb1
        
        gs = lbb['max_c'] + 1  # gap start col
        ge = rbb['min_c'] - 1  # gap end col
        gap = ge - gs + 1
        mid = gs + gap // 2
        lfe = mid - 1                                    # left fill end
        rfs = mid + 1 if gap % 2 == 1 else mid           # right fill start
        
        for r in range(rows):
            for c in range(gs, lfe + 1):
                grid[r][c] = lc
            for c in range(rfs, ge + 1):
                grid[r][c] = rc
    else:
        # Vertically separated
        if bb1['min_r'] < bb2['min_r']:
            tc, bc, tbb, bbb = c1, c2, bb1, bb2
        else:
            tc, bc, tbb, bbb = c2, c1, bb2, bb1
        
        gs = tbb['max_r'] + 1  # gap start row
        ge = bbb['min_r'] - 1  # gap end row
        gap = ge - gs + 1
        mid = gs + gap // 2
        tfe = mid - 1                                    # top fill end
        bfs = mid + 1 if gap % 2 == 1 else mid           # bottom fill start
        
        for r in range(gs, tfe + 1):
            for c in range(cols):
                grid[r][c] = tc
        for r in range(bfs, ge + 1):
            for c in range(cols):
                grid[r][c] = bc
    
    return grid


# === Test against all training examples ===
if __name__ == "__main__":
    train = [
        ([[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,1,1,4,4,4,4,4,4,4,4,4,4,6,6,4],[4,1,1,4,4,4,4,4,4,4,4,4,4,6,6,4],[4,1,1,4,4,4,4,4,4,4,4,4,4,6,6,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]],
         [[4,4,4,1,1,1,1,1,6,6,6,6,6,4,4,4],[4,1,1,1,1,1,1,1,6,6,6,6,6,6,6,4],[4,1,1,1,1,1,1,1,6,6,6,6,6,6,6,4],[4,1,1,1,1,1,1,1,6,6,6,6,6,6,6,4],[4,4,4,1,1,1,1,1,6,6,6,6,6,4,4,4]]),
        ([[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],[2,1,1,2,2,2,2,2,2,2,2,2,2,2,3,3,2],[2,1,1,2,2,2,2,2,2,2,2,2,2,2,3,3,2],[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]],
         [[2,2,2,1,1,1,1,1,2,3,3,3,3,3,2,2,2],[2,1,1,1,1,1,1,1,2,3,3,3,3,3,3,3,2],[2,1,1,1,1,1,1,1,2,3,3,3,3,3,3,3,2],[2,2,2,1,1,1,1,1,2,3,3,3,3,3,2,2,2]]),
        ([[8,8,8],[8,2,8],[8,2,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,1,8],[8,1,8],[8,8,8]],
         [[8,8,8],[8,2,8],[8,2,8],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[8,8,8],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[8,1,8],[8,1,8],[8,8,8]]),
    ]
    
    all_pass = True
    for i, (inp, exp) in enumerate(train):
        out = transform(inp)
        ok = out == exp
        print(f"Example {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            for r in range(len(exp)):
                if out[r] != exp[r]:
                    print(f"  Row {r}: got {out[r]}")
                    print(f"         exp {exp[r]}")
    
    # Test input
    test_in = [[2,2,2],[2,8,2],[2,8,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,1,2],[2,1,2],[2,2,2]]
    test_exp = [[2,2,2],[2,8,2],[2,8,2],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[8,8,8],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[2,1,2],[2,1,2],[2,2,2]]
    test_out = transform(test_in)
    test_ok = test_out == test_exp
    print(f"Test:      {'PASS' if test_ok else 'FAIL'}")
    
    print("\nSOLVED" if all_pass and test_ok else "\nFAILED")
