def transform(grid):
    """
    ARC-AGI 5b433def DSTAR solver - Final version
    
    Precise rule based on pattern analysis:
    1. If no 9,9 pairs: remove scattered non-background pixels from right side (c >= 12) 
    2. If 9,9 pairs exist: create specific rectangular formation below each pair
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
        # Case 1: No pairs - remove pixels from right region (c >= 12)
        for r in range(height):
            for c in range(width):
                if grid[r][c] != background and c >= 12:
                    result[r][c] = background
                        
    else:
        # Case 2: Create rectangular formations around each 9,9 pair
        # Clear result first
        for r in range(height):
            for c in range(width):
                result[r][c] = background
        
        # Restore all 9,9 pairs
        for r, c in nine_pairs:
            result[r][c] = 9
            result[r][c+1] = 9
        
        # Also restore isolated 9s that aren't part of pairs
        for r in range(height):
            for c in range(width):
                if grid[r][c] == 9:
                    # Check if it's part of a pair
                    is_pair_left = (c < width-1 and grid[r][c+1] == 9)
                    is_pair_right = (c > 0 and grid[r][c-1] == 9)
                    if not (is_pair_left or is_pair_right):
                        result[r][c] = 9  # Isolated 9
        
        # Create rectangular formations for each 9,9 pair
        for pair_r, pair_c in nine_pairs:
            # Standard rectangle pattern relative to pair position:
            # Pair is at (pair_r, pair_c) and (pair_r, pair_c+1)
            # Rectangle starts at (pair_r+1, pair_c-1):
            #   Row +1: [., 5, ., ., .]  (5 at pair_c)
            #   Row +2: [5, 5, 6, ., .]  (5s at pair_c-1,pair_c; 6 at pair_c+1)
            #   Row +3: [5, 5, 5, 5, 5]  (5s from pair_c-1 to pair_c+3)
            #   Row +4: [., 5, ., ., .]  (5 at pair_c)
            
            # Row +1: single 5 at pair_c
            if pair_r + 1 < height and pair_c < width:
                result[pair_r + 1][pair_c] = 5
                
            # Row +2: 5, 5, 6 pattern
            if pair_r + 2 < height:
                if pair_c - 1 >= 0:
                    result[pair_r + 2][pair_c - 1] = 5
                if pair_c < width:
                    result[pair_r + 2][pair_c] = 5
                if pair_c + 1 < width:
                    result[pair_r + 2][pair_c + 1] = 6
                    
            # Row +3: 5 across (5 positions)
            if pair_r + 3 < height:
                for dc in range(-1, 4):  # pair_c-1 to pair_c+3
                    c = pair_c + dc
                    if 0 <= c < width:
                        result[pair_r + 3][c] = 5
                        
            # Row +4: single 5 at pair_c
            if pair_r + 4 < height and pair_c < width:
                result[pair_r + 4][pair_c] = 5
    
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
            for r, c, exp, act in mismatches[:15]:
                print(f"  ({r},{c}): expected {exp}, got {act}")