def transform(grid):
    """
    ARC puzzle 4d567baa solver.
    
    Pattern: Count non-9 cells (N). Create a right-triangle tiled output.
    - Find smallest W where W+(W-2)+(W-4)+... >= N
    - Output is W*cols by W*rows
    - Tile row 0: all W tiles; row 1: W-2 tiles (right-aligned); row 2: W-4, etc.
    - When N=0 (all 9s), use W=6 and fill with 9s.
    """
    rows = len(grid)
    cols = len(grid[0])
    
    # Count non-9 cells
    n = sum(1 for r in grid for c in r if c != 9)
    
    # Special case: all 9s - use 6x6 tiling filled with 9s
    if n == 0:
        w = 6
        out_rows = w * rows
        out_cols = w * cols
        return [[9] * out_cols for _ in range(out_rows)]
    
    # Find minimum W where triangular sum >= n
    # Triangular sum with step 2: W + (W-2) + (W-4) + ... 
    def triangle_capacity(w):
        total = 0
        val = w
        while val > 0:
            total += val
            val -= 2
        return total
    
    w = 1
    while triangle_capacity(w) < n:
        w += 1
    
    # Create output grid filled with 9s
    out_rows = w * rows
    out_cols = w * cols
    output = [[9] * out_cols for _ in range(out_rows)]
    
    # Place tiles in triangle pattern
    tiles_placed = 0
    tile_row = 0
    tiles_in_row = w  # First row has W tiles
    start_col = 0     # First row starts at column 0
    
    while tiles_placed < n:
        # Calculate how many tiles to place in this row
        tiles_this_row = min(tiles_in_row, n - tiles_placed)
        
        # Place tiles from start_col to start_col + tiles_this_row - 1
        for tc in range(tiles_this_row):
            tile_col = start_col + tc
            # Copy input tile to output at (tile_row, tile_col)
            for r in range(rows):
                for c in range(cols):
                    out_r = tile_row * rows + r
                    out_c = tile_col * cols + c
                    output[out_r][out_c] = grid[r][c]
        
        tiles_placed += tiles_this_row
        tile_row += 1
        tiles_in_row -= 2  # Each row has 2 fewer tiles
        start_col += 2     # Each row starts 2 columns to the right
    
    return output


if __name__ == "__main__":
    import json
    
    # Load task
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['4d567baa']
    
    # Test on all training examples
    print("Testing on training examples:")
    all_passed = True
    for i, ex in enumerate(task['train']):
        result = transform(ex['input'])
        expected = ex['output']
        passed = result == expected
        all_passed = all_passed and passed
        print(f"  Example {i}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"    Expected size: {len(expected)}x{len(expected[0])}")
            print(f"    Got size: {len(result)}x{len(result[0])}")
    
    print(f"\nAll training examples: {'PASS' if all_passed else 'FAIL'}")
