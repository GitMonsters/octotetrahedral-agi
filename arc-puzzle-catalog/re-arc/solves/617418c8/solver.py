def transform(grid):
    """
    ARC puzzle 617418c8: Tile a pattern from edge across fill area.
    
    Rules:
    - Find pattern edge (row/col with multiple colors)
    - Find separator (adjacent row/col with single color)
    - Determine tile pattern based on position:
      - LEFT pattern: tile as-is horizontally
      - RIGHT pattern: rotate to start from fill color, tile horizontally
      - BOTTOM pattern: remove separator color, append at end, tile vertically
      - TOP pattern: tile as-is vertically
    """
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    result = grid.copy()
    
    def unique_count(arr):
        return len(set(arr.flatten()))
    
    def get_fill_color(region):
        from collections import Counter
        c = Counter(region.flatten())
        return c.most_common(1)[0][0]
    
    left_col = grid[:, 0]
    right_col = grid[:, -1]
    top_row = grid[0, :]
    bottom_row = grid[-1, :]
    
    # Determine which edge has the actual pattern by checking for:
    # 1. Multiple unique colors in the edge (excluding corners that belong to other edges)
    # 2. Adjacent separator row/col with single color
    # 3. Large fill area on opposite side
    # 4. PRIORITIZE by which edge has MORE unique colors (true pattern)
    
    def is_fill_area(region):
        """Check if region is mostly single color (fill)"""
        if region.size == 0:
            return False
        from collections import Counter
        c = Counter(region.flatten())
        most_common_count = c.most_common(1)[0][1]
        return most_common_count / region.size > 0.8
    
    # Compute all checks first
    left_col_unique = unique_count(left_col)
    col1_unique = unique_count(grid[:, 1]) if w > 1 else 0
    fill_right = is_fill_area(grid[:, 1:]) if w > 1 else False
    left_valid = left_col_unique > 1 and col1_unique == 1 and fill_right
    
    right_col_unique = unique_count(right_col)
    col_m2_unique = unique_count(grid[:, -2]) if w > 2 else 0
    fill_left = is_fill_area(grid[:, :-2]) if w > 2 else False
    right_valid = right_col_unique > 1 and col_m2_unique == 1 and fill_left
    
    bottom_row_unique = unique_count(bottom_row)
    row_m2_unique = unique_count(grid[-2, :]) if h > 2 else 0
    fill_above = is_fill_area(grid[:-2, :]) if h > 2 else False
    bottom_valid = h > 2 and bottom_row_unique > 1 and row_m2_unique == 1 and fill_above
    
    top_row_unique = unique_count(top_row)
    row1_unique = unique_count(grid[1, :]) if h > 1 else 0
    fill_below = is_fill_area(grid[2:, :]) if h > 2 else False
    top_valid = h > 2 and top_row_unique > 1 and row1_unique == 1 and fill_below
    
    # Find which valid pattern has the most unique colors
    candidates = []
    if left_valid:
        candidates.append(('left', left_col_unique))
    if right_valid:
        candidates.append(('right', right_col_unique))
    if bottom_valid:
        candidates.append(('bottom', bottom_row_unique))
    if top_valid:
        candidates.append(('top', top_row_unique))
    
    if not candidates:
        return result.tolist()
    
    # Choose the one with most unique colors
    best = max(candidates, key=lambda x: x[1])[0]
    
    if best == 'bottom':
        separator_color = grid[-2, 0]
        pattern = list(bottom_row)
        pattern_without_sep = [v for v in pattern if v != separator_color]
        tile_pattern = pattern_without_sep + [separator_color]
        for r in range(h - 2):
            idx = r % len(tile_pattern)
            for c in range(w):
                result[r, c] = tile_pattern[idx]
        return result.tolist()
    
    if best == 'top':
        separator_color = grid[1, 0]
        pattern = list(top_row)
        pattern_without_sep = [v for v in pattern if v != separator_color]
        tile_pattern = pattern_without_sep + [separator_color]
        for r in range(2, h):
            idx = (r - 2) % len(tile_pattern)
            for c in range(w):
                result[r, c] = tile_pattern[idx]
        return result.tolist()
    
    if best == 'right':
        fill_color = get_fill_color(grid[:, :-2])
        pattern = list(right_col)
        if fill_color in pattern:
            idx = pattern.index(fill_color)
            tile_pattern = pattern[idx:] + pattern[:idx]
        else:
            tile_pattern = pattern
        for c in range(w - 2):
            idx = c % len(tile_pattern)
            for r in range(h):
                result[r, c] = tile_pattern[idx]
        return result.tolist()
    
    if best == 'left':
        pattern = list(left_col)
        for c in range(2, w):
            idx = (c - 2) % len(pattern)
            for r in range(h):
                result[r, c] = pattern[idx]
        return result.tolist()
    
    return result.tolist()


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['617418c8']
    
    print("Testing on all training examples:\n")
    all_pass = True
    for i, ex in enumerate(task['train']):
        input_grid = ex['input']
        expected = ex['output']
        result = transform(input_grid)
        
        match = result == expected
        all_pass = all_pass and match
        
        print(f"Example {i}: {'✓ PASS' if match else '✗ FAIL'}")
        if not match:
            print(f"  Expected: {expected[:2]}...")
            print(f"  Got:      {result[:2]}...")
    
    print(f"\n{'All tests passed!' if all_pass else 'Some tests failed.'}")
