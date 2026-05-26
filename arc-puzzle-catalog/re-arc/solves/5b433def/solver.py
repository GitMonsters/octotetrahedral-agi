def transform(grid):
    """
    ARC-AGI 5b433def DSTAR solver - Retry 2
    
    Rule: 
    1. If no 9,9 pairs exist, remove scattered pixels from upper-right region
    2. If 9,9 pairs exist, create rectangular formations below/right of each pair
    """
    import copy
    
    result = copy.deepcopy(grid)
    height, width = len(grid), len(grid[0])
    
    # Detect background color
    color_counts = {}
    for r in range(height):
        for c in range(width):
            color = grid[r][c]
            color_counts[color] = color_counts.get(color, 0) + 1
    
    background = max(color_counts, key=color_counts.get)
    
    # Find horizontal 9,9 pairs
    nine_pairs = []
    for r in range(height):
        for c in range(width-1):
            if grid[r][c] == 9 and grid[r][c+1] == 9:
                nine_pairs.append((r, c))
    
    if len(nine_pairs) == 0:
        # Case 1: No pairs - remove pixels from upper-right area
        for r in range(height):
            for c in range(width):
                if grid[r][c] != background:
                    # Remove from upper-right region (c >= 12)
                    if c >= width * 2 // 3:
                        result[r][c] = background
                        
    else:
        # Case 2: Reorganize around 9,9 pairs
        # Clear result and preserve 9,9 pairs
        for r in range(height):
            for c in range(width):
                result[r][c] = background
        
        # Keep all 9,9 pairs
        for r, c in nine_pairs:
            result[r][c] = 9
            result[r][c+1] = 9
        
        # Collect all non-background, non-9 pixels
        other_pixels = []
        for r in range(height):
            for c in range(width):
                if grid[r][c] != background and grid[r][c] != 9:
                    other_pixels.append((grid[r][c], r, c))
        
        # Sort pixels by color for consistent placement
        other_pixels.sort()
        
        # Create rectangular formations for each 9,9 pair
        for pair_idx, (pair_r, pair_c) in enumerate(nine_pairs):
            # Each pair gets a rectangular formation in a specific pattern:
            # Rectangle starts 1 row below and slightly to the left/right
            
            rect_top = pair_r + 1
            rect_left = pair_c - 1
            
            # Standard rectangle pattern based on analysis:
            # Row 0: [5]
            # Row 1: [5, 5, 6]  
            # Row 2: [5, 5, 5, 5, 5]
            # Row 3: [_, 5]
            
            rectangle_pattern = [
                [0, 0],         # Row +1: one 5 at (pair_c-1, pair_r+1) 
                [0, 0, 1, 0],   # Row +2: 5s at -1,0 and +1, 6 at +1
                [0, 0, 0, 0, 0, 0], # Row +3: 5s from -1 to +3
                [3, 0],         # Row +4: 5 at +1
            ]
            
            # Get subset of pixels for this rectangle
            pixels_for_rect = other_pixels[pair_idx * 6:(pair_idx + 1) * 6] if pair_idx * 6 < len(other_pixels) else other_pixels[pair_idx * 3:]
            
            color_5_pixels = [p for p in pixels_for_rect if p[0] == 5]
            color_6_pixels = [p for p in pixels_for_rect if p[0] == 6]
            
            # Place rectangle pattern
            if rect_top + 3 < height and rect_left >= 0 and rect_left + 5 < width:
                # Row 1: single 5
                if color_5_pixels:
                    result[rect_top][rect_left + 1] = 5
                    
                # Row 2: 5, 5, 6
                if len(color_5_pixels) >= 2:
                    result[rect_top + 1][rect_left] = 5
                    result[rect_top + 1][rect_left + 1] = 5
                if color_6_pixels:
                    result[rect_top + 1][rect_left + 2] = 6
                    
                # Row 3: 5 across
                for dc in range(5):
                    if rect_left + dc < width and len(color_5_pixels) > dc + 2:
                        result[rect_top + 2][rect_left + dc] = 5
                        
                # Row 4: single 5 at position +1
                if len(color_5_pixels) >= 7:
                    result[rect_top + 3][rect_left + 1] = 5
    
    return result

solve = transform

# Test the solver
if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/apr12_tasks/5b433def.json', 'r') as f:
        task = json.load(f)
    
    print("Testing on training examples:")
    for i, example in enumerate(task['train']):
        print(f"\n=== Training Pair {i} ===")
        input_grid = example['input']
        expected_output = example['output']
        actual_output = transform(input_grid)
        
        # Compare outputs
        matches = True
        mismatches = []
        for r in range(len(expected_output)):
            for c in range(len(expected_output[0])):
                if expected_output[r][c] != actual_output[r][c]:
                    matches = False
                    mismatches.append((r, c, expected_output[r][c], actual_output[r][c]))
        
        if matches:
            print("✓ MATCH")
        else:
            print(f"✗ MISMATCH - {len(mismatches)} differences")
            for r, c, exp, act in mismatches[:10]:
                print(f"  ({r},{c}): expected {exp}, got {act}")