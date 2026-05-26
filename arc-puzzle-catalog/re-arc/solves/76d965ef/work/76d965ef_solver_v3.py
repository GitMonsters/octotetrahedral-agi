def solve(grid):
    """
    ARC-AGI Task 76d965ef: Fractal pattern transformation
    
    Based on detailed analysis, the rule is:
    1. Extract non-background pattern
    2. Create 2x input size output
    3. Tile the pattern in segments where each row has structure:
       [seg1, seg2, seg3] where segments are pattern-width sized
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
    
    # Key insight: The pattern creates fractal structure through recursive application
    # But the exact rule varies by case. Let me implement based on observed patterns.
    
    for r in range(out_h):
        row = []
        
        pattern_row_idx = r % pat_h
        current_pattern_row = pattern[pattern_row_idx]
        
        # Based on analysis, there are multiple segments per row
        num_complete_segments = out_w // pat_w
        remainder = out_w % pat_w
        
        if num_complete_segments == 3 and remainder == 0:
            # Case like Train 1: 18x18 with 6x6 pattern = 3 complete segments
            seg1 = pattern[0]  # Always first pattern row
            seg2 = pattern[0]  # Always first pattern row  
            seg3 = current_pattern_row  # Current pattern row
            row = seg1 + seg2 + seg3
            
        elif num_complete_segments == 2:
            # Cases like Train 0, Train 2: partial segments
            # From detailed analysis of Train 0:
            
            seg1 = pattern[0]  # First segment always pattern[0]
            
            if r < pat_h:
                # First cycle: use pattern structure
                seg2 = current_pattern_row
                # For remainder, continue the pattern
                remainder_part = current_pattern_row[:remainder] if remainder > 0 else []
            else:
                # Later cycles: different rule
                seg2 = pattern[0]  # Or some other rule
                remainder_part = [pattern[0][0]] * remainder if remainder > 0 else []
            
            row = seg1 + seg2 + remainder_part
            
        else:
            # Fallback: simple tiling
            for c in range(out_w):
                row.append(pattern[pattern_row_idx][c % pat_w])
        
        result.append(row[:out_w])  # Ensure correct width
    
    return result


# Let me try implementing the EXACT pattern I observed from Train 1 (which worked)
# But make it work for different segment counts

def solve_exact_pattern(grid):
    """
    Implement the exact pattern that worked for Train 1
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
        
        pattern_row_idx = r % pat_h
        
        # Build segments
        segments_needed = (out_w + pat_w - 1) // pat_w  # Ceiling division
        
        for seg_idx in range(segments_needed):
            if seg_idx == 0:
                # First segment: always pattern[0]
                segment = pattern[0]
            elif seg_idx == 1:
                # Second segment: always pattern[0]
                segment = pattern[0]
            else:
                # Later segments: use current pattern row
                segment = pattern[pattern_row_idx]
            
            # Add segment to row
            start_pos = seg_idx * pat_w
            for i, val in enumerate(segment):
                if start_pos + i < out_w:
                    row.append(val)
        
        result.append(row)
    
    return result

# Use this version
solve = solve_exact_pattern

# Test
def test_solve():
    import json
    
    with open('/Users/evanpieser/apr12_tasks/76d965ef.json', 'r') as f:
        data = json.load(f)
    
    print("=== TESTING EXACT PATTERN SOLVER ===")
    
    for i, example in enumerate(data['train']):
        input_grid = example['input']
        expected_output = example['output']
        predicted_output = solve(input_grid)
        
        matches = predicted_output == expected_output
        print(f"\nTrain {i}: {'PASS' if matches else 'FAIL'}")
        
        if not matches:
            print("Debugging...")
            # Check specific structure
            pat_w = len(predicted_output[0]) // 3 if len(predicted_output[0]) >= 18 else len(predicted_output[0]) // 2
            print(f"Pattern width estimate: {pat_w}")
            
            print("First row segments:")
            if len(predicted_output[0]) >= pat_w * 3:
                print(f"  Seg1: {predicted_output[0][:pat_w]}")
                print(f"  Seg2: {predicted_output[0][pat_w:pat_w*2]}")  
                print(f"  Seg3: {predicted_output[0][pat_w*2:pat_w*3]}")
                print(f"  Expected: {expected_output[0]}")
                
if __name__ == "__main__":
    test_solve()