def transform(grid):
    """
    Pattern: Grid has 2x2 cells separated by single-pixel grid lines.
    Find colored cells (non-background, non-grid), identify a template sequence,
    and propagate it based on anchor markers (typically color 7).
    """
    import copy
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)
    
    # Detect cell size and grid structure
    # Cells are 2x2 with 1-pixel grid lines between them
    cell_h = 2
    cell_w = 2
    sep = 1
    stride_r = cell_h + sep
    stride_c = cell_w + sep
    
    # Count colors to find background and grid line colors
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            color_counts[v] = color_counts.get(v, 0) + 1
    
    # Grid line color is the one appearing most in separator rows
    sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])
    grid_color = sorted_colors[0][0]  # Most frequent is grid lines
    bg_color = sorted_colors[1][0] if len(sorted_colors) > 1 else sorted_colors[0][0]  # Second most is background
    
    # Get cell value at cell coordinates (cr, cc)
    def get_cell(g, cr, cc):
        pr = cr * stride_r
        pc = cc * stride_c
        if pr < rows and pc < cols:
            return g[pr][pc]
        return None
    
    def set_cell(g, cr, cc, val):
        pr = cr * stride_r
        pc = cc * stride_c
        for dr in range(cell_h):
            for dc in range(cell_w):
                if pr + dr < rows and pc + dc < cols:
                    g[pr + dr][pc + dc] = val
    
    # Calculate number of cells
    num_cell_rows = (rows + sep) // stride_r
    num_cell_cols = (cols + sep) // stride_c
    
    # Find all special (non-background, non-grid) cells
    special_cells = {}  # (cr, cc) -> color
    for cr in range(num_cell_rows):
        for cc in range(num_cell_cols):
            val = get_cell(grid, cr, cc)
            if val is not None and val != bg_color and val != grid_color:
                special_cells[(cr, cc)] = val
    
    if not special_cells:
        return result
    
    # Group special cells by column to find template column
    cols_with_specials = {}
    for (cr, cc), val in special_cells.items():
        if cc not in cols_with_specials:
            cols_with_specials[cc] = []
        cols_with_specials[cc].append((cr, val))
    
    # Find the column(s) with the densest special coloring (template)
    # Template column has multiple different colors or most cells
    template_col = max(cols_with_specials.keys(), key=lambda c: len(cols_with_specials[c]))
    
    # Build template: relative offsets from anchor color (7 typically)
    # First, find which color acts as anchor (appears in multiple columns)
    anchor_color = 7  # Default
    color_col_counts = {}
    for (cr, cc), val in special_cells.items():
        if val not in color_col_counts:
            color_col_counts[val] = set()
        color_col_counts[val].add(cc)
    
    # Anchor color appears in multiple columns (or use 7 if present)
    for color, col_set in color_col_counts.items():
        if len(col_set) > 1 and color != 5 and color != 9:
            anchor_color = color
            break
    
    # Find template pattern: colors relative to anchor position in template column
    template_anchors = [(cr, val) for (cr, cc), val in special_cells.items() 
                        if cc == template_col and val == anchor_color]
    
    # Build pattern from ALL special cells in template region
    template_colors = {}  # row -> color (from template column or adjacent)
    for (cr, cc), val in special_cells.items():
        if abs(cc - template_col) <= 1:  # Template column or adjacent
            if cr not in template_colors or cc == template_col:
                template_colors[cr] = val
    
    # Now find anchor positions (color 7 or anchor) outside template region
    for (cr, cc), val in special_cells.items():
        if val == anchor_color and abs(cc - template_col) > 1:
            # Find corresponding anchor row in template
            anchor_rows_in_template = [r for r, v in template_colors.items() if v == anchor_color]
            if anchor_rows_in_template:
                template_anchor_row = anchor_rows_in_template[0]
                offset = cr - template_anchor_row
                
                # Apply template pattern with offset
                for trow, tcolor in template_colors.items():
                    new_row = trow + offset
                    if 0 <= new_row < num_cell_rows:
                        set_cell(result, new_row, cc, tcolor)
    
    # Also mark cells adjacent to colored cells with grid color (flood fill edges)
    # Re-scan to find all colored cells in result and mark adjacent cells
    new_special = {}
    for cr in range(num_cell_rows):
        for cc in range(num_cell_cols):
            val = get_cell(result, cr, cc)
            if val is not None and val != bg_color and val != grid_color:
                new_special[(cr, cc)] = val
    
    # For cells adjacent to colored cells, mark the surrounding grid with grid_color
    for (cr, cc), val in new_special.items():
        # Mark cells before this colored cell (to the left) in same row with grid_color
        # Actually, looking at the outputs, it seems like cells before colored sequence get grid_color
        pass  # Will handle this after checking basic pattern
    
    return result


if __name__ == "__main__":
    import json
    data = json.load(open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json'))
    puzzle = data['08cd3a60']
    
    all_pass = True
    for i, ex in enumerate(puzzle['train']):
        inp = ex['input']
        expected = ex['output']
        got = transform(inp)
        match = got == expected
        all_pass = all_pass and match
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"  Input shape: {len(inp)}x{len(inp[0])}")
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape: {len(got)}x{len(got[0])}")
    
    print(f"\nAll pass: {all_pass}")
