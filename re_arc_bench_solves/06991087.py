"""
ARC Puzzle 06991087 Solver

Pattern: There's a colored rectangle with a stripe passing through it.
The stripe extends beyond the rectangle asymmetrically. The transformation
moves the rectangle so the stripe extensions are balanced/symmetric on both sides.
"""

def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find background color (most common)
    unique, counts = np.unique(grid, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    
    # Find all non-background colors and their bounding boxes
    non_bg_colors = [c for c in unique if c != bg_color]
    
    if len(non_bg_colors) == 0:
        return grid.tolist()
    
    # Find rectangular regions for each color
    def get_bbox(color):
        positions = np.argwhere(grid == color)
        if len(positions) == 0:
            return None
        r_min, c_min = positions.min(axis=0)
        r_max, c_max = positions.max(axis=0)
        return r_min, c_min, r_max, c_max
    
    def is_filled_rect(color, bbox):
        r_min, c_min, r_max, c_max = bbox
        region = grid[r_min:r_max+1, c_min:c_max+1]
        return np.sum(region == color) == (r_max - r_min + 1) * (c_max - c_min + 1)
    
    # Identify the "rectangle" color (forms a filled rectangle possibly with holes from stripe)
    # and the "stripe" color (forms lines/stripes that pass through)
    rect_color = None
    stripe_color = None
    rect_bbox = None
    
    for color in non_bg_colors:
        bbox = get_bbox(color)
        if bbox is None:
            continue
        r_min, c_min, r_max, c_max = bbox
        region = grid[r_min:r_max+1, c_min:c_max+1]
        
        # Check if this forms a mostly-filled rectangle (might have stripe holes)
        total_cells = (r_max - r_min + 1) * (c_max - c_min + 1)
        fill_ratio = np.sum(region == color) / total_cells
        
        # Rectangle candidate: fills most of its bounding box
        if fill_ratio > 0.5:
            if rect_color is None or fill_ratio > 0.3:
                # Check if other color forms stripe through this region
                other_in_region = np.sum((region != color) & (region != bg_color))
                if other_in_region > 0:
                    rect_color = color
                    rect_bbox = bbox
    
    # Find the stripe color (the one passing through rectangle)
    if rect_color is not None:
        for color in non_bg_colors:
            if color != rect_color:
                stripe_color = color
                break
    
    if rect_color is None or stripe_color is None:
        # No transformation needed or pattern not recognized
        return grid.tolist()
    
    # Get the stripe bbox
    stripe_bbox = get_bbox(stripe_color)
    if stripe_bbox is None:
        return grid.tolist()
    
    r_min_r, c_min_r, r_max_r, c_max_r = rect_bbox
    r_min_s, c_min_s, r_max_s, c_max_s = stripe_bbox
    
    # Determine stripe orientation and asymmetry
    stripe_height = r_max_s - r_min_s + 1
    stripe_width = c_max_s - c_min_s + 1
    rect_height = r_max_r - r_min_r + 1
    rect_width = c_max_r - c_min_r + 1
    
    result = np.full_like(grid, bg_color)
    
    # Vertical stripe (tall and narrow relative to rectangle)
    if stripe_height > rect_height and stripe_width <= rect_width:
        # Calculate extensions above and below rectangle
        above = r_min_r - r_min_s  # stripe extends this much above rect
        below = r_max_s - r_max_r  # stripe extends this much below rect
        
        # Make symmetric: rectangle should have equal stripe extensions
        total_ext = above + below
        new_above = total_ext // 2
        new_below = total_ext - new_above
        
        # Calculate new rectangle position
        new_r_min_r = r_min_s + new_above
        new_r_max_r = new_r_min_r + rect_height - 1
        
        # Place the stripe
        for r in range(r_min_s, r_max_s + 1):
            for c in range(c_min_s, c_max_s + 1):
                if grid[r, c] == stripe_color:
                    result[r, c] = stripe_color
        
        # Place the rectangle (including stripe portions inside it)
        for dr in range(rect_height):
            for dc in range(rect_width):
                old_r = r_min_r + dr
                old_c = c_min_r + dc
                new_r = new_r_min_r + dr
                new_c = c_min_r + dc
                if 0 <= new_r < h and 0 <= new_c < w:
                    result[new_r, new_c] = grid[old_r, old_c]
    
    # Horizontal stripe (wide and short relative to rectangle)
    elif stripe_width > rect_width and stripe_height <= rect_height:
        # Calculate extensions left and right of rectangle
        left = c_min_r - c_min_s   # stripe extends this much left of rect
        right = c_max_s - c_max_r  # stripe extends this much right of rect
        
        # Make symmetric
        total_ext = left + right
        new_left = total_ext // 2
        new_right = total_ext - new_left
        
        # Calculate new rectangle position
        new_c_min_r = c_min_s + new_left
        new_c_max_r = new_c_min_r + rect_width - 1
        
        # Place the stripe
        for r in range(r_min_s, r_max_s + 1):
            for c in range(c_min_s, c_max_s + 1):
                if grid[r, c] == stripe_color:
                    result[r, c] = stripe_color
        
        # Place the rectangle
        for dr in range(rect_height):
            for dc in range(rect_width):
                old_r = r_min_r + dr
                old_c = c_min_r + dc
                new_r = r_min_r + dr
                new_c = new_c_min_r + dc
                if 0 <= new_r < h and 0 <= new_c < w:
                    result[new_r, new_c] = grid[old_r, old_c]
    else:
        # Check for multiple stripe segments (like train 2 with 3 separate rectangles)
        # These are already symmetric, return as-is
        return grid.tolist()
    
    return result.tolist()


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['06991087']
    
    print("Testing on all training examples:")
    all_pass = True
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"Train {i}: {status}")
        
        if not match:
            all_pass = False
            print(f"  Expected: {expected[:2]}...")
            print(f"  Got:      {result[:2]}...")
    
    print(f"\nAll tests passed: {all_pass}")
