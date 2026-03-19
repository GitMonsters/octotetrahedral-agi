"""
ARC Puzzle 284a499c Solver

Pattern: The grid has horizontal and vertical dividing lines that create a cross pattern.
Dividers are detected in two passes:
1. Find fully uniform rows/columns
2. Find rows/columns that are uniform except at intersections with the first pass dividers

The output extracts the region bounded by dividers, with:
1. The divider lines shown as borders
2. Interior cells showing a "waterfall fill" - for each column, non-background values 
   fill down from the top to the last occurrence of that value in the column.
"""

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the background color (most common value)
    all_vals = [v for row in grid for v in row]
    counts = {}
    for v in all_vals:
        counts[v] = counts.get(v, 0) + 1
    background = max(counts, key=counts.get)
    
    # Pass 1: Find fully uniform rows and columns
    uniform_row_set = set()
    h_dividers = []  # (row_idx, color)
    for r in range(rows):
        if len(set(grid[r])) == 1:
            uniform_row_set.add(r)
            h_dividers.append((r, grid[r][0]))
    
    uniform_col_set = set()
    v_dividers = []  # (col_idx, color)
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        if len(set(col_vals)) == 1:
            uniform_col_set.add(c)
            v_dividers.append((c, col_vals[0]))
    
    # Pass 2: Find columns that are uniform when excluding uniform rows
    for c in range(cols):
        if c in uniform_col_set:
            continue
        col_vals = [grid[r][c] for r in range(rows) if r not in uniform_row_set]
        if len(col_vals) > 0 and len(set(col_vals)) == 1:
            color = col_vals[0]
            if color != background:  # Only consider non-background dividers
                v_dividers.append((c, color))
                uniform_col_set.add(c)
    
    # Pass 2: Find rows that are uniform when excluding uniform columns
    for r in range(rows):
        if r in uniform_row_set:
            continue
        row_vals = [grid[r][c] for c in range(cols) if c not in uniform_col_set]
        if len(row_vals) > 0 and len(set(row_vals)) == 1:
            color = row_vals[0]
            if color != background:  # Only consider non-background dividers
                h_dividers.append((r, color))
                uniform_row_set.add(r)
    
    # Remove duplicates and sort
    h_dividers = sorted(list(set(h_dividers)))
    v_dividers = sorted(list(set(v_dividers)))
    
    # Get positions
    h_positions = [d[0] for d in h_dividers]
    v_positions = [d[0] for d in v_dividers]
    
    # Identify the middle region between dividers
    # Row region
    if len(h_positions) >= 2:
        mid_row_start = h_positions[0] + 1
        mid_row_end = h_positions[-1] - 1
    elif len(h_positions) == 1:
        pos = h_positions[0]
        before = pos
        after = rows - pos - 1
        if after <= before:
            mid_row_start = pos + 1
            mid_row_end = rows - 1
        else:
            mid_row_start = 0
            mid_row_end = pos - 1
    else:
        mid_row_start = 0
        mid_row_end = rows - 1
    
    # Column region
    if len(v_positions) >= 2:
        mid_col_start = v_positions[0] + 1
        mid_col_end = v_positions[-1] - 1
    elif len(v_positions) == 1:
        pos = v_positions[0]
        before = pos
        after = cols - pos - 1
        if after <= before:
            mid_col_start = pos + 1
            mid_col_end = cols - 1
        else:
            mid_col_start = 0
            mid_col_end = pos - 1
    else:
        mid_col_start = 0
        mid_col_end = cols - 1
    
    region_rows = mid_row_end - mid_row_start + 1
    region_cols = mid_col_end - mid_col_start + 1
    
    # Create ordered row elements based on the structure we want:
    # - If region is BETWEEN two h-dividers: h-div, region, h-div
    # - If region is AFTER the only h-divider: region, h-div (divider at bottom)
    # - If region is BEFORE the only h-divider: h-div, region (divider at top)
    row_elements = []  # ('div', input_row, color) or ('region', row_offset, input_row)
    
    if len(h_dividers) >= 2:
        # Two or more dividers - region is between them
        row_elements.append(('div', h_dividers[0][0], h_dividers[0][1]))
        for offset in range(region_rows):
            row_elements.append(('region', offset, mid_row_start + offset))
        row_elements.append(('div', h_dividers[-1][0], h_dividers[-1][1]))
    elif len(h_dividers) == 1:
        # One divider
        h_pos = h_dividers[0][0]
        if mid_row_start > h_pos:
            # Region is AFTER the divider -> region first, divider at bottom
            for offset in range(region_rows):
                row_elements.append(('region', offset, mid_row_start + offset))
            row_elements.append(('div', h_dividers[0][0], h_dividers[0][1]))
        else:
            # Region is BEFORE the divider -> divider at top, then region
            row_elements.append(('div', h_dividers[0][0], h_dividers[0][1]))
            for offset in range(region_rows):
                row_elements.append(('region', offset, mid_row_start + offset))
    else:
        # No dividers, just region
        for offset in range(region_rows):
            row_elements.append(('region', offset, mid_row_start + offset))
    
    # Create ordered column elements (same logic for columns)
    col_elements = []  # ('div', input_col, color) or ('region', col_offset, input_col)
    
    if len(v_dividers) >= 2:
        col_elements.append(('div', v_dividers[0][0], v_dividers[0][1]))
        for offset in range(region_cols):
            col_elements.append(('region', offset, mid_col_start + offset))
        col_elements.append(('div', v_dividers[-1][0], v_dividers[-1][1]))
    elif len(v_dividers) == 1:
        v_pos = v_dividers[0][0]
        if mid_col_start > v_pos:
            # Region is AFTER divider -> region first, divider at right
            for offset in range(region_cols):
                col_elements.append(('region', offset, mid_col_start + offset))
            col_elements.append(('div', v_dividers[0][0], v_dividers[0][1]))
        else:
            # Region is BEFORE divider -> divider at left, then region
            col_elements.append(('div', v_dividers[0][0], v_dividers[0][1]))
            for offset in range(region_cols):
                col_elements.append(('region', offset, mid_col_start + offset))
    else:
        for offset in range(region_cols):
            col_elements.append(('region', offset, mid_col_start + offset))
    
    # Pre-compute waterfall fill for each region column
    waterfall = {}  # col_offset -> (last_row_with_nonbg, nonbg_value)
    for col_offset in range(region_cols):
        inp_c = mid_col_start + col_offset
        last_idx = -1
        non_bg_val = background
        for row_offset in range(region_rows):
            inp_r = mid_row_start + row_offset
            if grid[inp_r][inp_c] != background:
                last_idx = row_offset
                non_bg_val = grid[inp_r][inp_c]
        waterfall[col_offset] = (last_idx, non_bg_val)
    
    # Build output
    output = []
    for row_elem in row_elements:
        row = []
        for col_elem in col_elements:
            if row_elem[0] == 'div':
                h_row = row_elem[1]
                h_color = row_elem[2]
                if col_elem[0] == 'div':
                    # Intersection - use actual grid value
                    row.append(grid[h_row][col_elem[1]])
                else:
                    # H-divider row, region column
                    # Use the h-div row value at this column position
                    col_offset = col_elem[1]
                    inp_c = mid_col_start + col_offset
                    val = grid[h_row][inp_c]
                    # If the h-div row at this position is the v-div color, use region bg instead
                    if val == background or any(val == d[1] for d in v_dividers):
                        # Check if val matches a v-div color that crosses this row
                        # Use the actual h-div row value unless it's the overall background
                        if h_color != background:
                            row.append(h_color)
                        else:
                            row.append(background)
                    else:
                        row.append(val)
            else:
                row_offset = row_elem[1]
                if col_elem[0] == 'div':
                    row.append(col_elem[2])
                else:
                    col_offset = col_elem[1]
                    last_idx, non_bg_val = waterfall[col_offset]
                    if last_idx >= 0 and row_offset <= last_idx:
                        row.append(non_bg_val)
                    else:
                        row.append(background)
        output.append(row)
    
    return output


if __name__ == "__main__":
    import json
    
    # Load the task
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['284a499c']
    
    print("Testing on all training examples:\n")
    all_pass = True
    
    for i, ex in enumerate(task['train']):
        input_grid = ex['input']
        expected = ex['output']
        result = transform(input_grid)
        
        match = result == expected
        all_pass = all_pass and match
        
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape: {len(result)}x{len(result[0])}")
            print(f"  Expected: {expected[:3]}...")
            print(f"  Got:      {result[:3]}...")
    
    print(f"\n{'All tests PASSED!' if all_pass else 'Some tests FAILED'}")
