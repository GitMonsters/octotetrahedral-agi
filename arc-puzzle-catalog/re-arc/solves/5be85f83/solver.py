def transform(grid):
    """
    Pattern: Find template rows/cols that have non-bg patterns.
    These templates define (selector_value, pattern) mappings.
    For each cell, look up its row and column selector values to get the output.
    """
    rows = len(grid)
    cols = len(grid[0])
    
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find selector column (vertical stripe)
    col_non_bg = []
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        unique_non_bg = len(set(v for v in col_vals if v != bg))
        col_non_bg.append((unique_non_bg, c))
    col_non_bg.sort(reverse=True)
    selector_col = col_non_bg[0][1]
    
    # Find selector row (horizontal stripe)
    row_non_bg = []
    for r in range(rows):
        unique_non_bg = len(set(v for v in grid[r] if v != bg))
        row_non_bg.append((unique_non_bg, r))
    row_non_bg.sort(reverse=True)
    selector_row = row_non_bg[0][1]
    
    # Get selector values
    row_selector = [grid[r][selector_col] for r in range(rows)]  # vertical stripe
    col_selector = grid[selector_row]  # horizontal stripe
    
    # Find template: rows that map row_selector_value -> row_pattern
    row_templates = {}
    for r in range(rows):
        key = row_selector[r]
        non_bg_count = sum(1 for v in grid[r] if v != bg)
        if non_bg_count > 1 and key not in row_templates:
            row_templates[key] = list(grid[r])
    
    # Find template: cols that map col_selector_value -> col_pattern
    col_templates = {}
    for c in range(cols):
        key = col_selector[c]
        col_vals = [grid[r][c] for r in range(rows)]
        non_bg_count = sum(1 for v in col_vals if v != bg)
        if non_bg_count > 1 and key not in col_templates:
            col_templates[key] = col_vals
    
    # Build output
    output = [[bg] * cols for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            row_key = row_selector[r]
            col_key = col_selector[c]
            
            # Try to get value from row template (tiling rows)
            if row_key in row_templates:
                output[r][c] = row_templates[row_key][c]
            # If row template doesn't help, try column template
            elif col_key in col_templates:
                output[r][c] = col_templates[col_key][r]
            else:
                output[r][c] = bg
    
    return output

# Test
if __name__ == "__main__":
    import json
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    puzzle = data['5be85f83']
    
    passed = 0
    for i, ex in enumerate(puzzle['train']):
        result = transform(ex['input'])
        expected = ex['output']
        if result == expected:
            print(f"Train {i}: PASS")
            passed += 1
        else:
            print(f"Train {i}: FAIL")
            print("Expected row 0:", expected[0])
            print("Got row 0:     ", result[0])
    print(f"\n{passed}/{len(puzzle['train'])} passed")
