"""
ARC Puzzle 468b6a14 Solver

Pattern: Lines extend across the grid. Where two different lines intersect, mark red (2).
- Find columns with non-background colors → extend vertically
- Find rows with non-background colors → extend horizontally  
- Intersections of different colored lines become red (2)
"""

def transform(grid):
    import copy
    from collections import Counter
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find the dominant non-background color in each column
    col_colors = {}
    for c in range(cols):
        colors = [grid[r][c] for r in range(rows) if grid[r][c] != bg]
        if colors:
            col_colors[c] = Counter(colors).most_common(1)[0][0]
    
    # Find the dominant non-background color in each row
    row_colors = {}
    for r in range(rows):
        colors = [grid[r][c] for c in range(cols) if grid[r][c] != bg]
        if colors:
            row_colors[r] = Counter(colors).most_common(1)[0][0]
    
    # Build output grid
    output = [[bg] * cols for _ in range(rows)]
    
    # First, fill in the extended lines
    for r in range(rows):
        for c in range(cols):
            # Check if this position is on an extended row or column line
            has_row = r in row_colors
            has_col = c in col_colors
            
            if has_row and has_col:
                # Intersection - mark red (2)
                output[r][c] = 2
            elif has_row:
                # Part of horizontal line
                output[r][c] = row_colors[r]
            elif has_col:
                # Part of vertical line
                output[r][c] = col_colors[c]
            else:
                output[r][c] = bg
    
    return output


if __name__ == "__main__":
    import json
    
    task = json.load(open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json'))['468b6a14']
    
    for i, ex in enumerate(task['train']):
        result = transform(ex['input'])
        expected = ex['output']
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            print("Expected:")
            for row in expected[:5]:
                print(row)
            print("Got:")
            for row in result[:5]:
                print(row)
