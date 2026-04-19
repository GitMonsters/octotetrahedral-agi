import json
from collections import Counter

def find_smallest_period(non_bg_cells, R, C):
    """Find smallest (P_r, P_c) consistent with non_bg_cells (no bg check)."""
    for P_row in range(1, R+1):
        for P_col in range(1, C+1):
            tile = {}; ok = True
            for r, c, v in non_bg_cells:
                k = (r % P_row, c % P_col)
                if k in tile and tile[k] != v: ok = False; break
                tile[k] = v
            if ok:
                return P_row, P_col, tile
    return None

def transform(grid):
    R, C = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    non_bg = [(r, c, grid[r][c]) for r in range(R) for c in range(C) if grid[r][c] != bg]
    
    # Step 1: try finding period from all non-bg cells
    result = find_smallest_period(non_bg, R, C)
    
    if result is None:
        # Step 2: try removing each non-bg color (the "noise" color)
        colors = sorted(set(v for _,_,v in non_bg), key=lambda v: -sum(1 for _,_,x in non_bg if x==v))
        for remove_color in colors:
            filtered = [(r,c,v) for r,c,v in non_bg if v != remove_color]
            result = find_smallest_period(filtered, R, C)
            if result is not None:
                break
    
    if result is None:
        return [row[:] for row in grid]
    
    P_row, P_col, tile = result
    out = [[bg]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            out[r][c] = tile.get((r%P_row, c%P_col), bg)
    return out

if __name__ == '__main__':
    with open('/Users/evanpieser/Desktop/ReArc45/re-arc_test_challenges-2026-04-05T23-26-25.json') as f:
        data = json.load(f)
    task = data['097428fa']
    passed = 0
    for i, ex in enumerate(task['train']):
        pred = transform([row[:] for row in ex['input']])
        ok = pred == ex['output']
        print(f'Train {i}: {"PASS" if ok else "FAIL"}')
        if ok: passed += 1
    print(f'{passed}/{len(task["train"])} passing')
