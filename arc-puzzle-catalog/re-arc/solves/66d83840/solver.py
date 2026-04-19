"""
Solver for ARC puzzle 66d83840

Pattern: Fractal/recursive tiling of nested L-shaped layers.
The innermost solid rectangle gets replaced with a copy of the whole input,
creating a diagonal tiling effect.
"""

def transform(grid: list[list[int]]) -> list[list[int]]:
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find the innermost rectangle (the "core") that can be tiled
    # This is the rectangular region that, when filled with copies of the input, creates the output
    
    # Strategy: Find where the pattern would tile by identifying the offset
    # The core rectangle is where all cells have the same value (background)
    
    # Find the dominant color (background) - usually the largest contiguous region
    # in one of the corners
    
    # Check each corner to find the "inner" rectangle position
    # The inner rect is the solid-colored region that gets replaced
    
    def find_core_rectangle(g):
        """Find the innermost solid rectangle that serves as the tiling seed."""
        h, w = g.shape
        
        # Try to find the rectangular region in each corner that's solid
        # Check corners: top-left, top-right, bottom-left, bottom-right
        corners = [
            (0, 0, 1, 1),      # top-left, grows down-right
            (0, w-1, 1, -1),   # top-right, grows down-left
            (h-1, 0, -1, 1),   # bottom-left, grows up-right
            (h-1, w-1, -1, -1) # bottom-right, grows up-left
        ]
        
        for start_r, start_c, dr, dc in corners:
            color = g[start_r, start_c]
            
            # Find max extent of this solid color region from corner
            max_r = 0
            max_c = 0
            
            # Find row extent
            for i in range(h):
                r = start_r + i * dr
                if 0 <= r < h and g[r, start_c] == color:
                    max_r = i + 1
                else:
                    break
            
            # Find col extent
            for j in range(w):
                c = start_c + j * dc
                if 0 <= c < w and g[start_r, c] == color:
                    max_c = j + 1
                else:
                    break
            
            # Check if this is a solid rectangle
            if max_r > 1 and max_c > 1:
                # Verify it's solid
                solid = True
                for i in range(max_r):
                    for j in range(max_c):
                        r = start_r + i * dr
                        c = start_c + j * dc
                        if g[r, c] != color:
                            solid = False
                            break
                    if not solid:
                        break
                
                if solid and max_r * max_c > 1:
                    # Return the rectangle info
                    if dr > 0:
                        r_start, r_end = start_r, start_r + max_r
                    else:
                        r_start, r_end = start_r - max_r + 1, start_r + 1
                    if dc > 0:
                        c_start, c_end = start_c, start_c + max_c
                    else:
                        c_start, c_end = start_c - max_c + 1, start_c + 1
                    
                    return (r_start, c_start, r_end, c_end, dr, dc)
        
        return None
    
    # Alternative approach: detect the tiling pattern by finding where copies fit
    # The output is created by tiling the input where the inner rectangle gets replaced
    
    def find_inner_rect(g):
        """Find the inner solid rectangle that will be replaced by copies."""
        h, w = g.shape
        
        # Look for L-shaped pattern - the inner corner shows where tiling happens
        # Find the largest corner rectangle of uniform color
        
        best = None
        best_area = 0
        
        for corner in range(4):
            if corner == 0:  # top-left
                r0, c0, dr, dc = 0, 0, 1, 1
            elif corner == 1:  # top-right
                r0, c0, dr, dc = 0, w-1, 1, -1
            elif corner == 2:  # bottom-left
                r0, c0, dr, dc = h-1, 0, -1, 1
            else:  # bottom-right
                r0, c0, dr, dc = h-1, w-1, -1, -1
            
            color = g[r0, c0]
            
            # Find the solid rectangle extent
            rect_h, rect_w = 1, 1
            
            # Extend in row direction
            for i in range(1, h):
                r = r0 + i * dr
                if 0 <= r < h and g[r, c0] == color:
                    rect_h = i + 1
                else:
                    break
            
            # Extend in col direction  
            for j in range(1, w):
                c = c0 + j * dc
                if 0 <= c < w and g[r0, c] == color:
                    rect_w = j + 1
                else:
                    break
            
            # Verify rectangle is solid
            is_solid = True
            for i in range(rect_h):
                for j in range(rect_w):
                    r = r0 + i * dr
                    c = c0 + j * dc
                    if g[r, c] != color:
                        is_solid = False
                        break
                if not is_solid:
                    break
            
            if is_solid:
                area = rect_h * rect_w
                if area > best_area:
                    best_area = area
                    if dr > 0:
                        rs, re = r0, r0 + rect_h
                    else:
                        rs, re = r0 - rect_h + 1, r0 + 1
                    if dc > 0:
                        cs, ce = c0, c0 + rect_w
                    else:
                        cs, ce = c0 - rect_w + 1, c0 + 1
                    best = (rs, cs, re, ce, dr, dc)
        
        return best
    
    rect_info = find_inner_rect(grid)
    
    if rect_info is None:
        return grid.tolist()
    
    rs, cs, re, ce, dr, dc = rect_info
    rect_h = re - rs
    rect_w = ce - cs
    
    # The tiling offset is the size of the inner rectangle
    # Output size: we need to expand until the inner rectangles fill
    
    # Calculate output dimensions based on tiling
    # The pattern tiles diagonally, each tile offset by (h - rect_h, w - rect_w) or similar
    
    # Determine tiling direction and offsets
    if dr > 0 and dc > 0:  # Inner rect at top-left
        offset_r = h - rect_h
        offset_c = w - rect_w
        # Count how many tiles fit
        n_tiles = min((h - rect_h), (w - rect_w))
        if n_tiles == 0:
            n_tiles = 1
        # Actually need to count based on where it ends
        n_r = (h + rect_h - 1) // rect_h if rect_h > 0 else 1
        n_c = (w + rect_w - 1) // rect_w if rect_w > 0 else 1
    elif dr > 0 and dc < 0:  # Inner rect at top-right
        offset_r = h - rect_h
        offset_c = -(w - rect_w)
    elif dr < 0 and dc > 0:  # Inner rect at bottom-left
        offset_r = -(h - rect_h)
        offset_c = w - rect_w
    else:  # Inner rect at bottom-right
        offset_r = -(h - rect_h)
        offset_c = -(w - rect_w)
    
    # Calculate number of tiles needed
    out_h = h * 2
    out_w = w * 2
    
    # Create output grid
    output = np.zeros((out_h, out_w), dtype=int)
    
    # Fill with tiled pattern
    # Each tile is placed at offset from where the inner rectangle would be
    step_r = h - rect_h
    step_c = w - rect_w
    
    # Determine starting position and direction
    if dr > 0:  # grows downward, so inner rect is at top, we tile downward
        start_r = 0
        dir_r = 1
    else:  # inner rect at bottom
        start_r = out_h - h
        dir_r = -1
    
    if dc > 0:  # grows rightward, inner rect at left
        start_c = 0
        dir_c = 1
    else:  # inner rect at right
        start_c = out_w - w
        dir_c = -1
    
    # Place tiles
    tiles_placed = 0
    positions = []
    
    # Generate tile positions
    cur_r, cur_c = start_r, start_c
    while 0 <= cur_r < out_h and 0 <= cur_c < out_w and cur_r + h <= out_h and cur_c + w <= out_w:
        positions.append((cur_r, cur_c))
        cur_r += step_r * dir_r
        cur_c += step_c * dir_c
        if step_r == 0 or step_c == 0:
            break
        tiles_placed += 1
        if tiles_placed > 10:  # Safety limit
            break
    
    # Fill output with background (use the inner rectangle color)
    bg_color = grid[rs, cs]
    output.fill(bg_color)
    
    # Place each tile
    for pr, pc in positions:
        output[pr:pr+h, pc:pc+w] = grid
    
    return output.tolist()


if __name__ == "__main__":
    import json
    
    # Load task
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['66d83840']
    
    print("Testing on all training examples:\n")
    
    all_passed = True
    for i, ex in enumerate(task['train']):
        input_grid = ex['input']
        expected = ex['output']
        result = transform(input_grid)
        
        passed = result == expected
        all_passed = all_passed and passed
        
        print(f"Example {i}: {'PASS' if passed else 'FAIL'}")
        print(f"  Input size: {len(input_grid)}x{len(input_grid[0])}")
        print(f"  Expected size: {len(expected)}x{len(expected[0])}")
        print(f"  Result size: {len(result)}x{len(result[0])}")
        
        if not passed:
            print("  Expected:")
            for row in expected[:5]:
                print(f"    {row}")
            if len(expected) > 5:
                print(f"    ... ({len(expected)} rows total)")
            print("  Got:")
            for row in result[:5]:
                print(f"    {row}")
            if len(result) > 5:
                print(f"    ... ({len(result)} rows total)")
        print()
    
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")
