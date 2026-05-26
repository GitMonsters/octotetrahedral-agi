def solve(grid):
    """
    ARC-AGI Task 76d965ef: Fractal doubling pattern
    
    Simple approach: Extract the non-background pattern and tile it perfectly
    """
    
    input_h, input_w = len(grid), len(grid[0])
    output_h, output_w = 2 * input_h, 2 * input_w
    
    # Find non-background pattern bounds
    min_r = max_r = min_c = max_c = None
    for r in range(input_h):
        for c in range(input_w):
            if grid[r][c] != 3:  # Not background
                if min_r is None:
                    min_r = max_r = r
                    min_c = max_c = c
                else:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
    
    # If no non-background pattern found, return scaled background
    if min_r is None:
        return [[3 for _ in range(output_w)] for _ in range(output_h)]
    
    # Extract pattern
    pattern = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(grid[r][c])
        pattern.append(row)
    
    pattern_h, pattern_w = len(pattern), len(pattern[0])
    
    # Create output by perfect tiling
    output = []
    for r in range(output_h):
        row = []
        for c in range(output_w):
            # Simple modular tiling
            pattern_r = r % pattern_h
            pattern_c = c % pattern_w
            row.append(pattern[pattern_r][pattern_c])
        output.append(row)
    
    return output


# Test the solver
if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/apr12_tasks/76d965ef.json', 'r') as f:
        data = json.load(f)
    
    print("=== TESTING SIMPLE TILING SOLVER ===")
    
    for i, pair in enumerate(data['train']):
        input_grid = pair['input']
        expected_output = pair['output']
        predicted_output = solve(input_grid)
        
        # Calculate accuracy
        correct = 0
        total = len(expected_output) * len(expected_output[0])
        
        if len(predicted_output) == len(expected_output) and len(predicted_output[0]) == len(expected_output[0]):
            for r in range(len(expected_output)):
                for c in range(len(expected_output[0])):
                    if predicted_output[r][c] == expected_output[r][c]:
                        correct += 1
        
        accuracy = correct / total * 100 if total > 0 else 0
        
        print(f"Train {i}: {accuracy:.1f}% accuracy")
        
        if accuracy == 100.0:
            print("  ✅ PERFECT MATCH!")
        else:
            print("  ❌ Still not perfect")
            
            # Show pattern info
            # Find pattern bounds
            min_r = max_r = min_c = max_c = None
            for r in range(len(input_grid)):
                for c in range(len(input_grid[0])):
                    if input_grid[r][c] != 3:
                        if min_r is None:
                            min_r = max_r = r
                            min_c = max_c = c
                        else:
                            min_r = min(min_r, r)
                            max_r = max(max_r, r)
                            min_c = min(min_c, c)
                            max_c = max(max_c, c)
            
            if min_r is not None:
                pattern_h = max_r - min_r + 1
                pattern_w = max_c - min_c + 1
                print(f"  Pattern: {pattern_h}x{pattern_w} at ({min_r},{min_c}) to ({max_r},{max_c})")
                print(f"  Output: {len(expected_output)}x{len(expected_output[0])}")
                print(f"  Tiles: {len(expected_output)/pattern_h:.1f} x {len(expected_output[0])/pattern_w:.1f}")
    
    print("\nNote: This simple approach may not capture the full complexity of the transformation.")