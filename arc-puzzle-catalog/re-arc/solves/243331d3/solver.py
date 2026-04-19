"""
ARC Puzzle 243331d3 Solver

Pattern: Markers form two perpendicular "lines" (one horizontal-ish, one vertical-ish).
These lines have gaps. The transformation:
1. Fills the rectangular region bounded by these lines with gray (5)
2. Extends lines from gaps in the boundary out to the grid edges
"""

def transform(grid):
    import copy
    grid = [list(row) for row in grid]
    h, w = len(grid), len(grid[0])
    
    # Find background color (most common)
    from collections import Counter
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    # Find marker positions
    markers = set((r, c) for r in range(h) for c in range(w) if grid[r][c] != bg)
    if not markers:
        return grid
    
    fill_color = 5  # Gray
    
    # Find rows with multiple markers (horizontal lines)
    row_marker_counts = {}
    for r, c in markers:
        row_marker_counts[r] = row_marker_counts.get(r, 0) + 1
    
    # Find cols with multiple markers (vertical lines)
    col_marker_counts = {}
    for r, c in markers:
        col_marker_counts[c] = col_marker_counts.get(c, 0) + 1
    
    # Find the main horizontal boundary rows (typically top and bottom of the figure)
    h_rows = sorted([r for r, cnt in row_marker_counts.items() if cnt >= 3])
    
    # Find the main vertical boundary cols
    v_cols = sorted([c for c, cnt in col_marker_counts.items() if cnt >= 2])
    
    # Get bounding box
    all_rows = sorted(set(r for r, c in markers))
    all_cols = sorted(set(c for r, c in markers))
    min_r, max_r = min(all_rows), max(all_rows)
    min_c, max_c = min(all_cols), max(all_cols)
    
    # Create output grid
    out = copy.deepcopy(grid)
    
    # Step 1: Fill between horizontal boundary lines
    # The two main horizontal rows define the interior
    if len(h_rows) >= 2:
        top_h_row = min(h_rows)
        bot_h_row = max(h_rows)
        
        # Get the column span of these rows
        top_cols = sorted([c for r, c in markers if r == top_h_row])
        bot_cols = sorted([c for r, c in markers if r == bot_h_row])
        
        h_min_c = min(min(top_cols), min(bot_cols))
        h_max_c = max(max(top_cols), max(bot_cols))
        
        # Fill interior rows between the horizontal boundaries
        for r in range(top_h_row + 1, bot_h_row):
            # Check if this row has markers on boundaries
            cols_in_row = sorted([c for mr, c in markers if mr == r])
            
            if cols_in_row:
                # Row has some markers - determine fill range
                row_min_c = min(cols_in_row)
                row_max_c = max(cols_in_row)
                
                # Fill from 0 to row_min_c if there are markers defining left boundary
                for c in range(0, row_max_c + 1):
                    if out[r][c] == bg:
                        out[r][c] = fill_color
                        
                # Also extend right if needed
                if row_max_c < h_max_c:
                    for c in range(row_max_c + 1, w):
                        if out[r][c] == bg:
                            out[r][c] = fill_color
            else:
                # No markers in this row - fill entire width of bounding box
                for c in range(0, w):
                    if out[r][c] == bg:
                        out[r][c] = fill_color
    
    # Step 2: Fill the horizontal boundary rows (gaps between markers)
    for hr in h_rows if len(h_rows) >= 2 else []:
        cols_in_row = sorted([c for r, c in markers if r == hr])
        if len(cols_in_row) >= 2:
            for c in range(min(cols_in_row), max(cols_in_row) + 1):
                if out[hr][c] == bg:
                    out[hr][c] = fill_color
    
    # Step 3: Find gaps in horizontal boundaries and extend vertically
    if len(h_rows) >= 2:
        top_h_row = min(h_rows)
        bot_h_row = max(h_rows)
        
        # Gaps in top row extend upward
        top_cols = sorted([c for r, c in markers if r == top_h_row])
        if top_cols:
            for c in range(min(top_cols), max(top_cols) + 1):
                if grid[top_h_row][c] == bg:  # Gap
                    for r in range(0, top_h_row):
                        out[r][c] = fill_color
        
        # Gaps in bottom row extend downward
        bot_cols = sorted([c for r, c in markers if r == bot_h_row])
        if bot_cols:
            for c in range(min(bot_cols), max(bot_cols) + 1):
                if grid[bot_h_row][c] == bg:  # Gap
                    for r in range(bot_h_row + 1, h):
                        out[r][c] = fill_color
    
    # Step 4: Fill vertical column lines (gaps between markers)
    for vc in v_cols:
        rows_in_col = sorted([r for r, c in markers if c == vc])
        if len(rows_in_col) >= 2:
            for r in range(min(rows_in_col), max(rows_in_col) + 1):
                if out[r][vc] == bg:
                    out[r][vc] = fill_color
    
    # Step 5: Find gaps in vertical boundaries and extend horizontally
    # The leftmost and rightmost vertical lines
    if v_cols:
        left_v_col = min(v_cols)
        right_v_col = max(v_cols)
        
        # Gaps in left column extend leftward
        left_rows = sorted([r for r, c in markers if c == left_v_col])
        if left_rows:
            for r in range(min(left_rows), max(left_rows) + 1):
                if grid[r][left_v_col] == bg:  # Gap
                    for c in range(0, left_v_col):
                        out[r][c] = fill_color
        
        # Gaps in right column extend rightward
        right_rows = sorted([r for r, c in markers if c == right_v_col])
        if right_rows:
            for r in range(min(right_rows), max(right_rows) + 1):
                if grid[r][right_v_col] == bg:  # Gap
                    for c in range(right_v_col + 1, w):
                        out[r][c] = fill_color
    
    return [list(row) for row in out]


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['243331d3']
    
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            print("Expected:")
            for row in expected:
                print(row)
            print("\nGot:")
            for row in result:
                print(row)
            print()
