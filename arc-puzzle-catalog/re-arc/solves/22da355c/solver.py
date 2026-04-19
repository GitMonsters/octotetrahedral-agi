import json
from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    nb = Counter(grid[r][c] for r in range(H) for c in range(W) if grid[r][c] != bg)
    if not nb:
        return grid
    big_color = nb.most_common(1)[0][0]
    
    # Find small cluster (all non-bg, non-big cells)
    sc = {(r, c): grid[r][c] for r in range(H) for c in range(W)
          if grid[r][c] != bg and grid[r][c] != big_color}
    if not sc:
        return grid
    
    sc_rows = [r for r, c in sc]; sc_cols = [c for r, c in sc]
    sr1, sr2 = min(sc_rows), max(sc_rows)
    sc1, sc2 = min(sc_cols), max(sc_cols)
    H_sc = sr2 - sr1 + 1; W_sc = sc2 - sc1 + 1
    
    # Big shape cells outside small cluster BB
    big_cells = {(r, c) for r in range(H) for c in range(W) if grid[r][c] == big_color}
    big_active = {(r, c) for r, c in big_cells if not (sr1 <= r <= sr2 and sc1 <= c <= sc2)}
    
    if not big_active:
        return grid
    
    bar1 = min(r for r, c in big_active); bar2 = max(r for r, c in big_active)
    bac1 = min(c for r, c in big_active); bac2 = max(c for r, c in big_active)
    H_ba = bar2 - bar1 + 1; W_ba = bac2 - bac1 + 1
    scale_r = H_ba / H_sc; scale_c = W_ba / W_sc
    
    result = []
    for i in range(H_sc):
        row = []
        for j in range(W_sc):
            r_start = bar1 + round(i * scale_r)
            r_end = bar1 + round((i + 1) * scale_r) - 1
            c_start = bac1 + round(j * scale_c)
            c_end = bac1 + round((j + 1) * scale_c) - 1
            has_big = any((r, c) in big_active
                         for r in range(r_start, r_end + 1)
                         for c in range(c_start, c_end + 1))
            if has_big:
                row.append(grid[sr1 + i][sc1 + j])
            else:
                row.append(bg)
        result.append(row)
    return result

if __name__ == '__main__':
    with open('/Users/evanpieser/Desktop/ReArc45/re-arc_test_challenges-2026-04-05T23-26-25.json') as f:
        data = json.load(f)
    task = data['22da355c']
    passed = 0
    for i, ex in enumerate(task['train']):
        pred = transform([row[:] for row in ex['input']])
        ok = pred == ex['output']
        print(f'Train {i}: {"PASS" if ok else "FAIL"}')
        if ok: passed += 1
    print(f'{passed}/{len(task["train"])} passing')
