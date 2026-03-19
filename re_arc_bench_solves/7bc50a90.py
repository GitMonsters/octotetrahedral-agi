"""
ARC Puzzle 7bc50a90 Solver

Pattern: Find marker lines (horizontal or vertical runs of a distinct color that span
a significant portion) and extract the rectangular region they define.
"""

def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find the background color (most common)
    unique, counts = np.unique(grid, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    
    # Find horizontal lines (consecutive same-color runs that are unusual)
    def find_horizontal_lines(grid, min_len=4):
        lines = []
        for r in range(h):
            row = grid[r]
            # Look for runs of non-bg color
            i = 0
            while i < w:
                if row[i] != bg_color:
                    color = row[i]
                    start = i
                    while i < w and row[i] == color:
                        i += 1
                    length = i - start
                    if length >= min_len:
                        lines.append((r, start, i - 1, color))
                else:
                    i += 1
        return lines
    
    # Find vertical lines
    def find_vertical_lines(grid, min_len=4):
        lines = []
        for c in range(w):
            col = grid[:, c]
            i = 0
            while i < h:
                if col[i] != bg_color:
                    color = col[i]
                    start = i
                    while i < h and col[i] == color:
                        i += 1
                    length = i - start
                    if length >= min_len:
                        lines.append((c, start, i - 1, color))
                else:
                    i += 1
        return lines
    
    horiz_lines = find_horizontal_lines(grid)
    vert_lines = find_vertical_lines(grid)
    
    # Check for horizontal line pairs (same color, same column range)
    if len(horiz_lines) >= 2:
        # Group by color
        from collections import defaultdict
        by_color = defaultdict(list)
        for r, c1, c2, color in horiz_lines:
            by_color[color].append((r, c1, c2))
        
        for color, lines in by_color.items():
            if len(lines) >= 2:
                # Find pairs with matching column ranges
                for i in range(len(lines)):
                    for j in range(i + 1, len(lines)):
                        r1, c1_1, c2_1 = lines[i]
                        r2, c1_2, c2_2 = lines[j]
                        # Check if columns overlap significantly
                        c_start = max(c1_1, c1_2)
                        c_end = min(c2_1, c2_2)
                        if c_end - c_start >= 3:  # significant overlap
                            row_start = min(r1, r2)
                            row_end = max(r1, r2)
                            # Use the overlapping column range
                            # But also check one cell before the run
                            col_start = min(c1_1, c1_2) - 1 if min(c1_1, c1_2) > 0 else min(c1_1, c1_2)
                            col_end = max(c2_1, c2_2) + 1 if max(c2_1, c2_2) < w - 1 else max(c2_1, c2_2)
                            return grid[row_start:row_end + 1, col_start:col_end + 1].tolist()
    
    # Check for vertical line pairs
    if len(vert_lines) >= 2:
        from collections import defaultdict
        by_color = defaultdict(list)
        for c, r1, r2, color in vert_lines:
            by_color[color].append((c, r1, r2))
        
        for color, lines in by_color.items():
            if len(lines) >= 2:
                for i in range(len(lines)):
                    for j in range(i + 1, len(lines)):
                        c1, r1_1, r2_1 = lines[i]
                        c2, r1_2, r2_2 = lines[j]
                        # Check if rows overlap significantly
                        r_start = max(r1_1, r1_2)
                        r_end = min(r2_1, r2_2)
                        if r_end - r_start >= 3:
                            col_start = min(c1, c2)
                            col_end = max(c1, c2)
                            row_start = min(r1_1, r1_2) - 1 if min(r1_1, r1_2) > 0 else min(r1_1, r1_2)
                            row_end = max(r2_1, r2_2) + 1 if max(r2_1, r2_2) < h - 1 else max(r2_1, r2_2)
                            return grid[row_start:row_end + 1, col_start:col_end + 1].tolist()
    
    # Fallback: look for rectangle defined by corner markers
    # Find non-background color that appears in corners of a rectangle
    non_bg_positions = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg_color:
                non_bg_positions.append((r, c, grid[r, c]))
    
    # Try to find 4 corners forming a rectangle
    from collections import defaultdict
    rows_by_col = defaultdict(set)
    cols_by_row = defaultdict(set)
    
    for r, c, color in non_bg_positions:
        rows_by_col[c].add(r)
        cols_by_row[r].add(c)
    
    # Find the smallest bounding rectangle that has markers at corners
    best_rect = None
    best_area = float('inf')
    
    colors_at_pos = {(r, c): color for r, c, color in non_bg_positions}
    
    for r1 in range(h):
        for r2 in range(r1 + 1, h):
            for c1 in range(w):
                for c2 in range(c1 + 1, w):
                    # Check if all 4 corners have the same non-bg color
                    corners = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                    corner_colors = [colors_at_pos.get(pos) for pos in corners]
                    if all(c is not None for c in corner_colors) and len(set(corner_colors)) == 1:
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area < best_area:
                            best_area = area
                            best_rect = (r1, r2, c1, c2)
    
    if best_rect:
        r1, r2, c1, c2 = best_rect
        return grid[r1:r2 + 1, c1:c2 + 1].tolist()
    
    return grid.tolist()


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['7bc50a90']
    
    print("Testing on all training examples:")
    all_correct = True
    for i, ex in enumerate(task['train']):
        result = transform(ex['input'])
        expected = ex['output']
        correct = result == expected
        all_correct = all_correct and correct
        print(f"\nTrain {i}: {'✓ PASS' if correct else '✗ FAIL'}")
        if not correct:
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape: {len(result)}x{len(result[0]) if result else 0}")
            print(f"  Expected: {expected[:2]}...")
            print(f"  Got:      {result[:2]}...")
    
    print(f"\n{'='*50}")
    print(f"Overall: {'ALL PASS' if all_correct else 'SOME FAILED'}")
