def transform(grid):
    """
    Puzzle 1999277e: Complete cross patterns formed by azure (8) cells.
    
    The 8s form partial crosses. We identify each cross by finding horizontal
    and vertical segments of 8s that intersect. Then fill in the missing cells
    with 0s to complete the cross arms.
    """
    import numpy as np
    
    grid = np.array(grid)
    result = grid.copy()
    rows, cols = grid.shape
    
    # Find all 8 positions
    eights = list(zip(*np.where(grid == 8)))
    if not eights:
        return result.tolist()
    
    # Find horizontal segments of 8s (allowing gaps)
    def find_horizontal_segments():
        segments = []
        for r in range(rows):
            row_eights = sorted([c for (rr, c) in eights if rr == r])
            if len(row_eights) >= 2:
                # Group nearby 8s into segments
                i = 0
                while i < len(row_eights):
                    start = row_eights[i]
                    end = start
                    j = i + 1
                    while j < len(row_eights) and row_eights[j] - end <= 4:
                        end = row_eights[j]
                        j += 1
                    if end > start:  # At least 2 cells span
                        segments.append((r, start, end))
                    i = j if j > i + 1 else i + 1
        return segments
    
    # Find vertical segments of 8s (allowing gaps)
    def find_vertical_segments():
        segments = []
        for c in range(cols):
            col_eights = sorted([r for (r, cc) in eights if cc == c])
            if len(col_eights) >= 2:
                i = 0
                while i < len(col_eights):
                    start = col_eights[i]
                    end = start
                    j = i + 1
                    while j < len(col_eights) and col_eights[j] - end <= 4:
                        end = col_eights[j]
                        j += 1
                    if end > start:
                        segments.append((c, start, end))
                    i = j if j > i + 1 else i + 1
        return segments
    
    h_segs = find_horizontal_segments()
    v_segs = find_vertical_segments()
    
    # Find crosses: horizontal and vertical segments that intersect
    crosses = []
    used_h = set()
    used_v = set()
    
    for hi, (h_row, h_start, h_end) in enumerate(h_segs):
        for vi, (v_col, v_start, v_end) in enumerate(v_segs):
            # Check if they intersect
            if h_start <= v_col <= h_end and v_start <= h_row <= v_end:
                # Intersection at (h_row, v_col)
                center_r, center_c = h_row, v_col
                
                # Calculate arm lengths from center
                arm_left = v_col - h_start
                arm_right = h_end - v_col
                arm_up = h_row - v_start
                arm_down = v_end - h_row
                
                # Make arms symmetric (use max length for each axis)
                h_arm = max(arm_left, arm_right)
                v_arm = max(arm_up, arm_down)
                
                crosses.append((center_r, center_c, h_arm, v_arm))
                used_h.add(hi)
                used_v.add(vi)
    
    # Draw the crosses with 0s
    for center_r, center_c, h_arm, v_arm in crosses:
        # Horizontal arm
        for dc in range(-h_arm, h_arm + 1):
            c = center_c + dc
            if 0 <= c < cols and result[center_r, c] != 8:
                result[center_r, c] = 0
        
        # Vertical arm
        for dr in range(-v_arm, v_arm + 1):
            r = center_r + dr
            if 0 <= r < rows and result[r, center_c] != 8:
                result[r, center_c] = 0
    
    return result.tolist()


if __name__ == "__main__":
    import json
    
    # Load task
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['1999277e']
    
    print("Testing on all training examples:\n")
    all_passed = True
    
    for i, example in enumerate(task['train']):
        input_grid = example['input']
        expected = example['output']
        predicted = transform(input_grid)
        
        match = predicted == expected
        all_passed = all_passed and match
        
        print(f"Train {i}: {'✓ PASS' if match else '✗ FAIL'}")
        
        if not match:
            import numpy as np
            pred_arr = np.array(predicted)
            exp_arr = np.array(expected)
            diff = pred_arr != exp_arr
            diff_count = np.sum(diff)
            print(f"  Differences: {diff_count} cells")
            diff_positions = list(zip(*np.where(diff)))[:5]
            for r, c in diff_positions:
                print(f"    ({r},{c}): predicted={pred_arr[r,c]}, expected={exp_arr[r,c]}")
    
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")
