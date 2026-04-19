import json
from collections import Counter

def find_max_solid_rect(grid, bg):
    R, C = len(grid), len(grid[0])
    heights = [0] * C
    best = (0, 0, 0, 0, 0)
    for r in range(R):
        for c in range(C):
            heights[c] = heights[c] + 1 if grid[r][c] != bg else 0
        stack = []
        for c in range(C + 1):
            h = heights[c] if c < C else 0
            start = c
            while stack and stack[-1][1] > h:
                idx, height = stack.pop()
                area = height * (c - idx)
                if area > best[0]:
                    best = (area, r - height + 1, r, idx, idx + c - idx - 1)
                start = idx
            stack.append((start, h))
    return best[1:]

def transform(grid):
    R, C = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    non_bg_color = next((v for row in grid for v in row if v != bg), None)
    non_bg_pos = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != bg]
    if not non_bg_pos:
        return [row[:] for row in grid]
    
    cr0, cr1, cc0, cc1 = find_max_solid_rect(grid, bg)
    core_set = {(r, c) for r in range(cr0, cr1+1) for c in range(cc0, cc1+1)}
    outliers = [(r, c) for r, c in non_bg_pos if (r, c) not in core_set]
    
    right_rows = {r for r, c in outliers if cr0 <= r <= cr1 and c > cc1}
    left_rows  = {r for r, c in outliers if cr0 <= r <= cr1 and c < cc0}
    above_out  = [(r, c) for r, c in outliers if r < cr0]
    below_out  = [(r, c) for r, c in outliers if r > cr1]
    
    out_cells = set(core_set)
    for r in right_rows: out_cells.add((r, cc1 + 1))
    for r in left_rows:  out_cells.add((r, cc0 - 1))
    
    if out_cells:
        ext_left = min(c for _, c in out_cells)
        ext_right = max(c for _, c in out_cells)
    else:
        ext_left, ext_right = cc0, cc1
    
    def map_col(c):
        if c < ext_left: return ext_left - 1
        if c > ext_right: return ext_right
        return c
    
    if above_out:
        for r, c in above_out: out_cells.add((cr0 - 1, map_col(c)))
    if below_out:
        for r, c in below_out: out_cells.add((cr1 + 1, map_col(c)))
    
    out = [[bg] * C for _ in range(R)]
    for r, c in out_cells:
        if 0 <= r < R and 0 <= c < C:
            out[r][c] = non_bg_color
    return out

if __name__ == '__main__':
    with open('/Users/evanpieser/Desktop/ReArc45/re-arc_test_challenges-2026-04-05T23-26-25.json') as f:
        data = json.load(f)
    task = data['12216db6']
    passed = 0
    for i, ex in enumerate(task['train']):
        pred = transform([row[:] for row in ex['input']])
        ok = pred == ex['output']
        print(f'Train {i}: {"PASS" if ok else "FAIL"}')
        if ok: passed += 1
    print(f'{passed}/{len(task["train"])} passing')
