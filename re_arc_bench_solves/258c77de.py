def transform(grid):
    """
    ARC Puzzle 258c77de:
    - Grid is divided into cells by divider lines (rows/cols of uniform color)
    - Mark top-left cell with color 9
    - Mark center cell with color 7
    - Mark bottom-right cell with color 1
    """
    import copy
    
    grid = [list(row) for row in grid]
    h, w = len(grid), len(grid[0])
    
    # Find divider color - the color that forms complete row/column lines
    # It's typically the most common color in full rows/cols
    def is_uniform_row(r):
        return len(set(grid[r])) == 1
    
    def is_uniform_col(c):
        return len(set(grid[r][c] for r in range(h))) == 1
    
    # Find all uniform rows and columns
    uniform_rows = [r for r in range(h) if is_uniform_row(r)]
    uniform_cols = [c for c in range(w) if is_uniform_col(c)]
    
    # Determine divider color from uniform lines
    divider_color = None
    if uniform_rows:
        divider_color = grid[uniform_rows[0]][0]
    elif uniform_cols:
        divider_color = grid[0][uniform_cols[0]]
    
    # Find cell boundaries - divider rows/cols must have divider_color
    if divider_color is not None:
        div_rows = [r for r in uniform_rows if grid[r][0] == divider_color]
        div_cols = [c for c in uniform_cols if grid[0][c] == divider_color]
    else:
        div_rows = []
        div_cols = []
    
    # Build cell row ranges
    row_starts = [0]
    for r in div_rows:
        if r > row_starts[-1]:
            row_starts.append(r + 1)
    row_ranges = []
    for i, start in enumerate(row_starts):
        if i + 1 < len(row_starts):
            end = row_starts[i + 1] - 2  # exclude the divider
        else:
            end = h - 1
        # Find actual end before next divider
        end = start
        while end + 1 < h and (end + 1 not in div_rows):
            end += 1
        if start <= end:
            row_ranges.append((start, end))
    
    # Build cell column ranges
    col_starts = [0]
    for c in div_cols:
        if c > col_starts[-1]:
            col_starts.append(c + 1)
    col_ranges = []
    for i, start in enumerate(col_starts):
        end = start
        while end + 1 < w and (end + 1 not in div_cols):
            end += 1
        if start <= end:
            col_ranges.append((start, end))
    
    # If no cells found, treat entire grid as one cell
    if not row_ranges:
        row_ranges = [(0, h - 1)]
    if not col_ranges:
        col_ranges = [(0, w - 1)]
    
    num_row_cells = len(row_ranges)
    num_col_cells = len(col_ranges)
    
    # Get cell coordinates
    def get_cell(row_idx, col_idx):
        r_start, r_end = row_ranges[row_idx]
        c_start, c_end = col_ranges[col_idx]
        return r_start, r_end, c_start, c_end
    
    # Fill cell with color (only non-divider pixels)
    def fill_cell(row_idx, col_idx, color):
        r_start, r_end, c_start, c_end = get_cell(row_idx, col_idx)
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                if divider_color is None or grid[r][c] != divider_color:
                    grid[r][c] = color
    
    # Mark top-left cell with 9
    fill_cell(0, 0, 9)
    
    # Mark center cell with 7
    center_row = num_row_cells // 2
    center_col = num_col_cells // 2
    fill_cell(center_row, center_col, 7)
    
    # Mark bottom-right cell with 1
    fill_cell(num_row_cells - 1, num_col_cells - 1, 1)
    
    return [list(row) for row in grid]


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['258c77de']
    
    all_pass = True
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        all_pass = all_pass and match
        print(f"Example {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            print("Expected:")
            for row in expected[:5]:
                print(row)
            print("Got:")
            for row in result[:5]:
                print(row)
            print()
    
    print(f"\nAll training examples: {'PASS' if all_pass else 'FAIL'}")
