"""
ARC Puzzle 3b4c1a4b Solver

Pattern: Lines (horizontal and vertical) form shapes. Where a horizontal line 
intersects a rectangular box (bounded region), fill the interior of that
bounded region with color 8 (azure).

The key insight: Find rectangles bounded by the line color where:
- There's a clear rectangular boundary (top, bottom, left, right edges in line color)
- The interior contains background color that should be filled with 8
"""

def transform(grid):
    import copy
    grid = [list(row) for row in grid]
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common) and line color
    from collections import Counter
    all_colors = [c for row in grid for c in row]
    color_counts = Counter(all_colors)
    bg_color = color_counts.most_common(1)[0][0]
    
    # Line color is the second most common
    line_color = None
    for color, count in color_counts.most_common():
        if color != bg_color:
            line_color = color
            break
    
    if line_color is None:
        return grid
    
    result = copy.deepcopy(grid)
    
    # Find bounded rectangular regions and fill interiors
    # Strategy: For each cell that is bg_color, check if it's enclosed by line_color rectangle
    
    filled = [[False] * cols for _ in range(rows)]
    
    # Try all possible rectangles defined by line boundaries
    for top in range(rows):
        for left in range(cols):
            if grid[top][left] != line_color:
                continue
            
            for bottom in range(top + 2, rows):
                for right in range(left + 2, cols):
                    if is_bounded_rectangle(grid, top, left, bottom, right, line_color, bg_color):
                        # Fill interior with 8
                        for r in range(top + 1, bottom):
                            for c in range(left + 1, right):
                                if grid[r][c] == bg_color:
                                    result[r][c] = 8
    
    return result


def is_bounded_rectangle(grid, top, left, bottom, right, line_color, bg_color):
    """
    Check if this is a valid bounded rectangle:
    - All four corners must be line_color
    - All four edges must be entirely line_color
    - Interior must have at least one bg_color cell
    """
    rows = len(grid)
    cols = len(grid[0])
    
    if bottom >= rows or right >= cols:
        return False
    
    # Check all four corners
    if grid[top][left] != line_color:
        return False
    if grid[top][right] != line_color:
        return False
    if grid[bottom][left] != line_color:
        return False
    if grid[bottom][right] != line_color:
        return False
    
    # Check top edge - all must be line_color
    for c in range(left, right + 1):
        if grid[top][c] != line_color:
            return False
    
    # Check bottom edge
    for c in range(left, right + 1):
        if grid[bottom][c] != line_color:
            return False
    
    # Check left edge
    for r in range(top, bottom + 1):
        if grid[r][left] != line_color:
            return False
    
    # Check right edge
    for r in range(top, bottom + 1):
        if grid[r][right] != line_color:
            return False
    
    # Interior must have at least one background cell
    has_bg = False
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            if grid[r][c] == bg_color:
                has_bg = True
                break
        if has_bg:
            break
    
    return has_bg


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['3b4c1a4b']
    
    print("Testing on all training examples:\n")
    all_passed = True
    
    for i, ex in enumerate(task['train']):
        input_grid = ex['input']
        expected = ex['output']
        result = transform(input_grid)
        
        match = result == expected
        all_passed = all_passed and match
        
        print(f"Training Example {i}: {'✓ PASS' if match else '✗ FAIL'}")
        
        if not match:
            print("Expected:")
            for row in expected:
                print(row)
            print("Got:")
            for row in result:
                print(row)
            print()
    
    print(f"\nOverall: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")
