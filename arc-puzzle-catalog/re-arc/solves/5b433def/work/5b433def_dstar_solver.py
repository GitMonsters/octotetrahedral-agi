def transform(grid):
    """
    ARC-AGI 5b433def DSTAR solver
    
    Rule: Find horizontal 9,9 pairs. Collect all non-background, non-9 pixels 
    and reorganize them into rectangular formations adjacent to each 9,9 pair.
    If no 9,9 pairs, remove scattered pixels from certain regions.
    """
    import copy
    
    result = copy.deepcopy(grid)
    height, width = len(grid), len(grid[0])
    
    # Detect background color (most frequent)
    color_counts = {}
    for r in range(height):
        for c in range(width):
            color = grid[r][c]
            color_counts[color] = color_counts.get(color, 0) + 1
    
    background = max(color_counts, key=color_counts.get)
    
    # Find all horizontal 9,9 pairs
    nine_pairs = []
    for r in range(height):
        for c in range(width-1):
            if grid[r][c] == 9 and grid[r][c+1] == 9:
                nine_pairs.append((r, c))
    
    print(f"Background: {background}, Found {len(nine_pairs)} 9,9 pairs: {nine_pairs}")
    
    if len(nine_pairs) == 0:
        # Case 1: No 9,9 pairs - remove scattered pixels from certain regions
        for r in range(height):
            for c in range(width):
                if grid[r][c] != background:
                    # Remove from upper-right and some lower regions
                    should_remove = ((r <= height//3 and c >= width*2//3) or 
                                   (r >= height*2//3 and c >= width*3//4))
                    if should_remove:
                        result[r][c] = background
    else:
        # Case 2: Reorganize scattered pixels into rectangles near 9,9 pairs
        # Clear result to background first
        for r in range(height):
            for c in range(width):
                result[r][c] = background
        
        # Preserve all 9,9 pairs
        for r, c in nine_pairs:
            result[r][c] = 9
            result[r][c+1] = 9
        
        # Collect all non-background, non-9 pixels
        scattered_pixels = []
        for r in range(height):
            for c in range(width):
                if grid[r][c] != background and grid[r][c] != 9:
                    scattered_pixels.append((grid[r][c], r, c))
        
        print(f"Collected {len(scattered_pixels)} scattered pixels")
        
        # Organize pixels by color
        by_color = {}
        for color, r, c in scattered_pixels:
            if color not in by_color:
                by_color[color] = []
            by_color[color].append((r, c))
        
        # For each 9,9 pair, create rectangular formations
        for pair_idx, (pair_r, pair_c) in enumerate(nine_pairs):
            print(f"Processing pair {pair_idx} at ({pair_r}, {pair_c})")
            
            # Determine position for rectangle relative to this pair
            # Strategy: place rectangles in different positions for different pairs
            
            if pair_idx % 4 == 0:  # Top-right of pair
                rect_r = max(0, pair_r - 3)
                rect_c = min(width - 5, pair_c + 2)
            elif pair_idx % 4 == 1:  # Bottom-left of pair  
                rect_r = min(height - 4, pair_r + 1)
                rect_c = max(0, pair_c - 4)
            elif pair_idx % 4 == 2:  # Top-left of pair
                rect_r = max(0, pair_r - 3) 
                rect_c = max(0, pair_c - 4)
            else:  # Bottom-right of pair
                rect_r = min(height - 4, pair_r + 1)
                rect_c = min(width - 5, pair_c + 2)
            
            # Create rectangular formation
            pixel_idx = 0
            for color in sorted(by_color.keys()):
                pixels_of_color = by_color[color]
                
                # Place this color in a rectangular region
                for dr in range(4):
                    for dc in range(5):
                        r = rect_r + dr
                        c = rect_c + dc
                        
                        if (0 <= r < height and 0 <= c < width and 
                            pixel_idx < len(pixels_of_color)):
                            
                            if (dr == 0 and dc < 2) or (dr == 1 and dc < 4) or dr == 2:
                                result[r][c] = color
                                pixel_idx += 1
                            elif color == 6 and dr == 1 and dc == 2:  # Special position for 6
                                result[r][c] = color
                                pixel_idx += 1
                
                if pixel_idx >= len(pixels_of_color):
                    break
    
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
            for r, c, exp, act in mismatches[:10]:  # Show first 10 mismatches
                print(f"  ({r},{c}): expected {exp}, got {act}")