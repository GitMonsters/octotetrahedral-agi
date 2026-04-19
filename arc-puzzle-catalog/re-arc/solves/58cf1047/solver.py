"""
ARC Puzzle 58cf1047 Solver

Pattern: There's a vertical (or horizontal) stripe dividing the grid. On one 
side, there's a diagonal line of marker pixels (may be same color as stripe).
The transformation extends this diagonal with gray (5) pixels toward the stripe
until reaching the stripe boundary.
"""

def transform(grid):
    import copy
    grid = [list(row) for row in grid]
    height = len(grid)
    width = len(grid[0])
    output = copy.deepcopy(grid)
    
    # Find the background color (most common)
    color_count = {}
    for row in grid:
        for c in row:
            color_count[c] = color_count.get(c, 0) + 1
    background = max(color_count, key=color_count.get)
    
    # Find the stripe - contiguous region of non-background color
    # Check for vertical stripe first
    stripe_cols = set()
    stripe_rows = set()
    stripe_color = None
    
    # Find columns that are entirely (or mostly) non-background
    for c in range(width):
        col_vals = [grid[r][c] for r in range(height)]
        non_bg = [v for v in col_vals if v != background]
        if len(non_bg) == height:  # entire column is stripe
            stripe_cols.add(c)
            if stripe_color is None:
                stripe_color = non_bg[0]
    
    # If no vertical stripe, check for horizontal stripe
    if not stripe_cols:
        for r in range(height):
            row_vals = grid[r]
            non_bg = [v for v in row_vals if v != background]
            if len(non_bg) == width:  # entire row is stripe
                stripe_rows.add(r)
                if stripe_color is None:
                    stripe_color = non_bg[0]
    
    # Determine stripe boundaries
    if stripe_cols:
        stripe_min_c = min(stripe_cols)
        stripe_max_c = max(stripe_cols)
        is_vertical = True
    elif stripe_rows:
        stripe_min_r = min(stripe_rows)
        stripe_max_r = max(stripe_rows)
        is_vertical = False
    else:
        return output
    
    # Find diagonal marker pixels (non-background, outside stripe)
    markers = []
    for r in range(height):
        for c in range(width):
            if grid[r][c] != background:
                if is_vertical and c not in stripe_cols:
                    markers.append((r, c))
                elif not is_vertical and r not in stripe_rows:
                    markers.append((r, c))
    
    if len(markers) < 2:
        return output
    
    # Sort markers to find diagonal direction
    markers.sort()
    
    # Determine diagonal direction from markers
    dr = markers[1][0] - markers[0][0]  # row direction
    dc = markers[1][1] - markers[0][1]  # col direction
    
    # Normalize direction
    if dr != 0:
        dr = dr // abs(dr)
    if dc != 0:
        dc = dc // abs(dc)
    
    first_marker = markers[0]
    last_marker = markers[-1]
    
    # Extend toward the stripe from the marker endpoint closer to stripe
    if is_vertical:
        stripe_center = (stripe_min_c + stripe_max_c) / 2
        # Determine which direction goes toward stripe
        if first_marker[1] > stripe_center:
            # Markers are to the right, extend leftward (toward stripe)
            ext_dr, ext_dc = dr, -abs(dc)
        else:
            # Markers are to the left, extend rightward (toward stripe)
            ext_dr, ext_dc = dr, abs(dc)
        
        # Start from the endpoint closest to stripe
        if dc > 0:  # markers going right, so last_marker is rightmost
            if first_marker[1] > stripe_center:
                start = first_marker  # extend from left side (first)
            else:
                start = last_marker  # extend from right side (last)
        else:  # markers going left
            if first_marker[1] > stripe_center:
                start = last_marker
            else:
                start = first_marker
        
        # Actually: extend from the end that points toward the stripe
        # If markers go from upper-right to lower-left (dc < 0), and stripe is on left
        # then extend from lower-left (last marker) toward stripe
        
        # Simpler approach: extend from last marker in marker direction
        # AND extend from first marker in opposite direction
        # The one that approaches stripe will fill, the other won't
        
        def in_stripe(c):
            return stripe_min_c <= c <= stripe_max_c
        
        # Extend from last marker continuing the diagonal
        r, c = last_marker[0] + dr, last_marker[1] + dc
        while 0 <= r < height and 0 <= c < width:
            if in_stripe(c):
                break
            if output[r][c] == background:
                output[r][c] = 5
            r += dr
            c += dc
        
        # Extend from first marker in opposite direction
        r, c = first_marker[0] - dr, first_marker[1] - dc
        while 0 <= r < height and 0 <= c < width:
            if in_stripe(c):
                break
            if output[r][c] == background:
                output[r][c] = 5
            r -= dr
            c -= dc
    else:
        # Horizontal stripe
        stripe_center = (stripe_min_r + stripe_max_r) / 2
        
        def in_stripe(r):
            return stripe_min_r <= r <= stripe_max_r
        
        # Extend from last marker continuing the diagonal
        r, c = last_marker[0] + dr, last_marker[1] + dc
        while 0 <= r < height and 0 <= c < width:
            if in_stripe(r):
                break
            if output[r][c] == background:
                output[r][c] = 5
            r += dr
            c += dc
        
        # Extend from first marker in opposite direction
        r, c = first_marker[0] - dr, first_marker[1] - dc
        while 0 <= r < height and 0 <= c < width:
            if in_stripe(r):
                break
            if output[r][c] == background:
                output[r][c] = 5
            r -= dr
            c -= dc
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['58cf1047']
    
    print("Testing on all training examples:\n")
    all_passed = True
    
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        passed = result == expected
        all_passed = all_passed and passed
        
        print(f"Example {i}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"  Expected first 3 rows: {expected[:3]}")
            print(f"  Got first 3 rows: {result[:3]}")
    
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
