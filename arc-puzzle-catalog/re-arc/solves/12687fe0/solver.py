import json
from collections import Counter

def transform(grid):
    R, C = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    
    bg_cols = [c for c in range(C) if all(grid[r][c] == bg for r in range(R))]
    bg_rows = [r for r in range(R) if all(grid[r][c] == bg for c in range(C))]
    
    sep_cols = [-1] + bg_cols + [C]
    sep_rows = [-1] + bg_rows + [R]
    bands_c = [(sep_cols[i]+1, sep_cols[i+1]) for i in range(len(sep_cols)-1)
               if sep_cols[i+1] > sep_cols[i]+1]
    bands_r = [(sep_rows[i]+1, sep_rows[i+1]) for i in range(len(sep_rows)-1)
               if sep_rows[i+1] > sep_rows[i]+1]
    
    out = []
    for rs, re in bands_r:
        row = []
        for cs, ce in bands_c:
            vals = [grid[r][c] for r in range(rs, re) for c in range(cs, ce)]
            row.append(Counter(vals).most_common(1)[0][0])
        out.append(row)
    return out

if __name__ == '__main__':
    with open('/Users/evanpieser/Desktop/ReArc45/re-arc_test_challenges-2026-04-05T23-26-25.json') as f:
        data = json.load(f)
    task = data['12687fe0']
    passed = 0
    for i, ex in enumerate(task['train']):
        pred = transform([row[:] for row in ex['input']])
        ok = pred == ex['output']
        print(f'Train {i}: {"PASS" if ok else "FAIL"}')
        if ok: passed += 1
    print(f'{passed}/{len(task["train"])} passing')
