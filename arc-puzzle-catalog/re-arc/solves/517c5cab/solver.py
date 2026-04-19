def transform(grid):
    """
    ARC puzzle 517c5cab:
    - Find a divider stripe (horizontal or vertical band of uniform color)
    - Find marker pixels (third color scattered in background areas)  
    - Extend divider color to cover positions where markers exist
    """
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find the divider stripe (contiguous rows or columns of same color)
    def find_divider():
        # Check for horizontal stripe
        for start_row in range(h):
            if len(set(grid[start_row])) == 1:
                stripe_color = grid[start_row, 0]
                end_row = start_row
                while end_row + 1 < h and len(set(grid[end_row + 1])) == 1 and grid[end_row + 1, 0] == stripe_color:
                    end_row += 1
                if end_row - start_row >= 1:  # At least 2 rows
                    return 'horizontal', stripe_color, start_row, end_row
        
        # Check for vertical stripe
        for start_col in range(w):
            col_vals = grid[:, start_col]
            if len(set(col_vals)) == 1:
                stripe_color = col_vals[0]
                end_col = start_col
                while end_col + 1 < w and len(set(grid[:, end_col + 1])) == 1 and grid[0, end_col + 1] == stripe_color:
                    end_col += 1
                if end_col - start_col >= 1:  # At least 2 cols
                    return 'vertical', stripe_color, start_col, end_col
        
        return None, None, None, None
    
    orientation, divider_color, start, end = find_divider()
    
    if orientation is None:
        return grid.tolist()
    
    # Find all unique colors
    unique_colors = set(grid.flatten())
    
    # Find background color (most common non-divider color outside stripe)
    if orientation == 'horizontal':
        outside = np.vstack([grid[:start, :], grid[end+1:, :]])
    else:
        outside = np.hstack([grid[:, :start], grid[:, end+1:]])
    
    color_counts = {}
    for c in unique_colors:
        if c != divider_color:
            color_counts[c] = np.sum(outside == c)
    
    background_color = max(color_counts, key=color_counts.get) if color_counts else None
    
    # Marker colors are everything else (not divider, not background)
    marker_colors = unique_colors - {divider_color, background_color}
    
    result = grid.copy()
    
    # For each marker, extend divider color to cover it
    if orientation == 'horizontal':
        # Horizontal stripe at rows [start, end]
        # Extend vertically (up and down)
        for row in range(h):
            for col in range(w):
                if grid[row, col] in marker_colors:
                    # Draw line from stripe to this marker
                    if row < start:
                        # Marker is above stripe, extend from row to start-1
                        for r in range(row, start):
                            result[r, col] = divider_color
                    elif row > end:
                        # Marker is below stripe, extend from end+1 to row
                        for r in range(end + 1, row + 1):
                            result[r, col] = divider_color
    else:
        # Vertical stripe at columns [start, end]
        # Extend horizontally (left and right)
        for row in range(h):
            for col in range(w):
                if grid[row, col] in marker_colors:
                    # Draw line from stripe to this marker
                    if col < start:
                        # Marker is left of stripe, extend from col to start-1
                        for c in range(col, start):
                            result[row, c] = divider_color
                    elif col > end:
                        # Marker is right of stripe, extend from end+1 to col
                        for c in range(end + 1, col + 1):
                            result[row, c] = divider_color
    
    return result.tolist()


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['517c5cab']
    
    print("Testing on all training examples:\n")
    all_pass = True
    
    for i, ex in enumerate(task['train']):
        result = transform(ex['input'])
        expected = ex['output']
        passed = result == expected
        all_pass = all_pass and passed
        
        print(f"Example {i}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape: {len(result)}x{len(result[0])}")
            # Find differences
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  Diff at ({r},{c}): expected {expected[r][c]}, got {result[r][c]}")
    
    print(f"\n{'All tests passed!' if all_pass else 'Some tests failed.'}")
