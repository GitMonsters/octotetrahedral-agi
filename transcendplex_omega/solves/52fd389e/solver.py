import json
import numpy as np
from scipy import ndimage


def solve(grid):
    """
    For each connected region of 4s:
    1. Find marking color (non-zero, non-4 cell in region)
    2. Count number of marked cells
    3. Expand region by count cells in all directions
    4. Fill expanded border with marking color
    """
    grid = np.array(grid)
    output = grid.copy().astype(int)
    h, w = grid.shape
    
    # Find connected components of 4s
    labeled, num_features = ndimage.label(grid == 4)
    
    for region_id in range(1, num_features + 1):
        region_mask = labeled == region_id
        fours_pos = np.where(region_mask)
        
        r_min, r_max = fours_pos[0].min(), fours_pos[0].max()
        c_min, c_max = fours_pos[1].min(), fours_pos[1].max()
        
        # Find marking color and count marked cells
        region = grid[r_min:r_max+1, c_min:c_max+1]
        marks = region[(region != 0) & (region != 4)]
        
        if len(marks) == 0:
            continue
        
        marking_color = int(marks[0])
        mark_count = len(marks)
        
        # Expand by mark_count cells in all directions
        border_r_min = max(0, r_min - mark_count)
        border_r_max = min(h - 1, r_max + mark_count)
        border_c_min = max(0, c_min - mark_count)
        border_c_max = min(w - 1, c_max + mark_count)
        
        # Fill border with marking color (except where 4s are)
        for r in range(border_r_min, border_r_max + 1):
            for c in range(border_c_min, border_c_max + 1):
                if grid[r, c] != 4:
                    output[r, c] = marking_color
    
    return output.tolist()


if __name__ == "__main__":
    task_path = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/52fd389e.json"
    
    with open(task_path, "r") as f:
        task = json.load(f)
    
    print("Testing training examples...")
    all_pass = True
    
    for idx, example in enumerate(task["train"]):
        input_grid = example["input"]
        expected_output = example["output"]
        predicted_output = solve(input_grid)
        
        match = np.array_equal(predicted_output, expected_output)
        status = "PASS" if match else "FAIL"
        print(f"  Example {idx}: {status}")
        
        if not match:
            all_pass = False
            print(f"    Expected shape: {np.array(expected_output).shape}")
            print(f"    Got shape: {np.array(predicted_output).shape}")
    
    print("\n" + ("="*40))
    if all_pass:
        print("ALL TRAINING EXAMPLES PASS")
    else:
        print("SOME EXAMPLES FAILED")
