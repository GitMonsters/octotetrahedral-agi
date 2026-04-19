def transform(grid):
    grid = [list(row) for row in grid]
    H, W = len(grid), len(grid[0])
    from collections import Counter
    
    # Find separator rows and columns
    sep_rows = [r for r in range(H) if len(set(grid[r])) == 1]
    sep_cols = [c for c in range(W) if len(set(grid[r][c] for r in range(H))) == 1]
    
    # Get cell ranges
    def get_ranges(seps, total):
        ranges = []
        seps = sorted(set(seps))
        prev = 0
        for s in seps:
            if s > prev:
                ranges.append((prev, s))
            prev = s + 1
        if prev < total:
            ranges.append((prev, total))
        return ranges
    
    row_ranges = get_ranges(sep_rows, H)
    col_ranges = get_ranges(sep_cols, W)
    
    # Find separator and background colors
    sep_color = grid[sep_rows[0]][0] if sep_rows else (grid[0][sep_cols[0]] if sep_cols else None)
    flat = [c for row in grid for c in row]
    non_sep = [c for c in flat if c != sep_color]
    bg_color = Counter(non_sep).most_common(1)[0][0] if non_sep else Counter(flat).most_common(1)[0][0]
    
    # Get cell coordinates for a pixel position
    def get_cell(r, c):
        cell_row = cell_col = local_row = local_col = 0
        for ri, (rs, re) in enumerate(row_ranges):
            if rs <= r < re:
                cell_row = ri
                local_row = r - rs
                break
        for ci, (cs, ce) in enumerate(col_ranges):
            if cs <= c < ce:
                cell_col = ci
                local_col = c - cs
                break
        return cell_row, cell_col, local_row, local_col
    
    # Find all marker colors and their positions
    markers_by_color = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != bg_color and v != sep_color:
                if v not in markers_by_color:
                    markers_by_color[v] = []
                cell_row, cell_col, local_row, local_col = get_cell(r, c)
                markers_by_color[v].append((cell_row, cell_col, local_row, local_col))
    
    # For each color, compute bounding box of cells and fill
    result = [row[:] for row in grid]
    
    for color, positions in markers_by_color.items():
        if not positions:
            continue
        
        # Group by local position
        by_local = {}
        for cell_row, cell_col, local_row, local_col in positions:
            key = (local_row, local_col)
            if key not in by_local:
                by_local[key] = []
            by_local[key].append((cell_row, cell_col))
        
        # For each local position group, find bounding box and fill
        for (local_row, local_col), cells in by_local.items():
            min_cr = min(cr for cr, cc in cells)
            max_cr = max(cr for cr, cc in cells)
            min_cc = min(cc for cr, cc in cells)
            max_cc = max(cc for cr, cc in cells)
            
            # Fill all cells in bounding box
            for cr in range(min_cr, max_cr + 1):
                for cc in range(min_cc, max_cc + 1):
                    # Get actual pixel position
                    r_start, r_end = row_ranges[cr]
                    c_start, c_end = col_ranges[cc]
                    
                    if local_row < (r_end - r_start) and local_col < (c_end - c_start):
                        result[r_start + local_row][c_start + local_col] = color
    
    return [list(row) for row in result]


if __name__ == "__main__":
    import json
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    puzzle = data['197c2823']
    
    all_pass = True
    for i, ex in enumerate(puzzle['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            all_pass = False
            # Show differences
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  Diff at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
    
    print(f"\nAll passed: {all_pass}")
