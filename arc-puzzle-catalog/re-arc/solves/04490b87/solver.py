import json
from collections import Counter

def transform(grid):
    R, C = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    non_bg = [(r, c, v) for r, row in enumerate(grid) for c, v in enumerate(row) if v != bg]
    
    out = [[bg] * C for _ in range(R)]
    if not non_bg:
        return out
    
    # Single pixel: determine if at col-edge or row-edge
    # If at col edge: fill entire row. If at row edge: fill entire column.
    def at_row_edge(r): return r == 0 or r == R - 1
    def at_col_edge(c): return c == 0 or c == C - 1
    
    # Determine extension direction
    # If any pixel is at a col edge -> row extension (horizontal stripes)
    # If any pixel is at a row edge -> col extension (vertical stripes)
    col_edge_pix = [(r, c, v) for r, c, v in non_bg if at_col_edge(c)]
    row_edge_pix = [(r, c, v) for r, c, v in non_bg if at_row_edge(r)]
    
    if col_edge_pix:
        # Extend as row stripes: fill entire rows
        # Sort pixels by row and extend with step = |row_diff|
        pix_sorted = sorted(non_bg, key=lambda x: x[0])
        rows_filled = {}
        for r, c, v in non_bg:
            rows_filled[r] = v
        
        if len(non_bg) >= 2:
            rows_list = sorted(rows_filled.keys())
            step = rows_list[1] - rows_list[0] if len(rows_list) >= 2 else 1
            if step > 0:
                # Extend forward from min row
                r = min(rows_list)
                idx = 0
                while r < R:
                    v = rows_filled.get(r, rows_filled[rows_list[idx % len(rows_list)]])
                    for col in range(C):
                        out[r][col] = v
                    r += step
                    idx = (idx + 1) % len(rows_list)
        else:
            r, c, v = non_bg[0]
            for col in range(C):
                out[r][col] = v
    elif row_edge_pix:
        # Extend as column stripes: fill entire columns
        pix_sorted = sorted(non_bg, key=lambda x: x[1])
        cols_filled = {}
        for r, c, v in non_bg:
            cols_filled[c] = v
        
        if len(non_bg) >= 2:
            cols_list = sorted(cols_filled.keys())
            step = cols_list[1] - cols_list[0] if len(cols_list) >= 2 else 1
            if step > 0:
                col = min(cols_list)
                idx = 0
                while col < C:
                    v = cols_filled.get(col, cols_filled[cols_list[idx % len(cols_list)]])
                    for row in range(R):
                        out[row][col] = v
                    col += step
                    idx = (idx + 1) % len(cols_list)
        else:
            r, c, v = non_bg[0]
            for row in range(R):
                out[row][c] = v
    
    return out

if __name__ == '__main__':
    with open('/Users/evanpieser/Desktop/ReArc45/re-arc_test_challenges-2026-04-05T23-26-25.json') as f:
        data = json.load(f)
    task = data['04490b87']
    passed = 0
    for i, ex in enumerate(task['train']):
        pred = transform([row[:] for row in ex['input']])
        ok = pred == ex['output']
        print(f'Train {i}: {"PASS" if ok else "FAIL"}')
        if ok: passed += 1
    print(f'{passed}/{len(task["train"])} passing')
