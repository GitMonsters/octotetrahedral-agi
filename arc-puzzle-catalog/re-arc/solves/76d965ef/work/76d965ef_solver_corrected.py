def solve(grid):
    """
    ARC-AGI Task 76d965ef: Fractal pattern transformation
    
    FINAL CORRECT RULE discovered:
    - Segment 1: Always pattern[0]
    - Segment 2: Always pattern[0] 
    - Segment 3: 
      - First cycle (r < pat_h): pattern[r % pat_h]
      - Later cycles (r >= pat_h): Fill with last element of pattern[r % pat_h]
    """
    input_h, input_w = len(grid), len(grid[0])
    
    # Find pattern bounds
    min_r = max_r = min_c = max_c = None
    for r in range(input_h):
        for c in range(input_w):
            if grid[r][c] != 3:
                if min_r is None:
                    min_r = max_r = r
                    min_c = max_c = c
                else:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
    
    if min_r is None:
        return [[3] * (input_w * 2) for _ in range(input_h * 2)]
    
    # Extract pattern
    pattern = []
    for r in range(min_r, max_r + 1):
        row = [grid[r][c] for c in range(min_c, max_c + 1)]
        pattern.append(row)
    
    pat_h, pat_w = len(pattern), len(pattern[0])
    out_h, out_w = input_h * 2, input_w * 2
    
    result = []
    
    for r in range(out_h):
        row = []
        
        # Determine how many segments we need
        segments_needed = (out_w + pat_w - 1) // pat_w
        
        pattern_row_idx = r % pat_h
        
        for seg_idx in range(segments_needed):
            if seg_idx == 0:
                # Segment 1: Always pattern[0]
                segment = pattern[0]
            elif seg_idx == 1:
                # Segment 2: Always pattern[0]
                segment = pattern[0]
            elif seg_idx == 2:
                # Segment 3: The key insight
                if r < pat_h:
                    # First cycle: use pattern[row % pat_h]
                    segment = pattern[pattern_row_idx]
                else:
                    # Later cycles: fill with last element of pattern[row % pat_h]
                    last_element = pattern[pattern_row_idx][-1]
                    segment = [last_element] * pat_w
            else:
                # Additional segments (if any): use pattern[row % pat_h]
                segment = pattern[pattern_row_idx]
            
            # Add segment to row (truncate if needed)
            for i, val in enumerate(segment):
                if len(row) < out_w:
                    row.append(val)
        
        result.append(row)
    
    return result


# Test function
def test_solve():
    import json
    
    with open('/Users/evanpieser/apr12_tasks/76d965ef.json', 'r') as f:
        data = json.load(f)
    
    print("=== TESTING FINAL CORRECT SOLVER ===")
    
    all_pass = True
    for i, example in enumerate(data['train']):
        input_grid = example['input']
        expected_output = example['output']
        predicted_output = solve(input_grid)
        
        matches = predicted_output == expected_output
        print(f"\nTrain {i}: {'PASS' if matches else 'FAIL'}")
        
        if not matches:
            all_pass = False
            if i <= 1:  # Debug first two
                print("First difference:")
                for j in range(min(3, len(expected_output))):
                    if expected_output[j] != predicted_output[j]:
                        print(f"  Row {j}:")
                        print(f"    Expected:  {expected_output[j]}")
                        print(f"    Predicted: {predicted_output[j]}")
                        break
    
    print(f"\nOVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass

if __name__ == "__main__":
    test_solve()