import numpy as np
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    Solve ARC puzzle 8cb8642d.
    
    Pattern: For each rectangle filled with a single color that contains exactly
    ONE cell with a different color, fill the inner region with 0s and draw diagonal 
    lines from the 4 corners of the inner region toward the center using the 
    special color. The rectangle's border remains the same.
    """
    grid = np.array(grid)
    result = grid.copy()
    
    # Find all unique non-zero values in the grid
    unique_vals = np.unique(grid)
    unique_vals = unique_vals[unique_vals != 0]
    
    # Process each rectangular region
    for val in unique_vals:
        # Find cells with this value
        mask = grid == val
        rows, cols = np.where(mask)
        
        if len(rows) == 0:
            continue
        
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()
        
        # Extract the rectangle
        rect = grid[min_r:max_r+1, min_c:max_c+1]
        height, width = rect.shape
        
        # Find special values (values that are not the fill value and not 0)
        # Count how many special cells there are
        special_val = None
        special_cell_count = 0
        
        for r in range(height):
            for c in range(width):
                if rect[r, c] != val and rect[r, c] != 0:
                    if special_val is None:
                        special_val = rect[r, c]
                    special_cell_count += 1
        
        # Only transform if there's exactly ONE special cell
        if special_val is not None and special_cell_count == 1 and height > 2 and width > 2:
            # Create the output for this rectangle
            output_rect = rect.copy()
            
            # Fill inner region (excluding border) with 0s
            output_rect[1:-1, 1:-1] = 0
            
            # Calculate the center of the inner region
            inner_height = height - 2
            inner_width = width - 2
            center_r = (inner_height - 1) / 2.0
            center_c = (inner_width - 1) / 2.0
            
            # Draw diagonal lines from the 4 corners of the inner region to the center
            # Corners in inner coordinates
            corners_inner = [
                (0, 0),
                (0, inner_width - 1),
                (inner_height - 1, 0),
                (inner_height - 1, inner_width - 1)
            ]
            
            for corner_r, corner_c in corners_inner:
                # Draw line from corner towards center
                r, c = corner_r, corner_c
                
                # Direction to move towards center
                dr = 1 if center_r > r else (-1 if center_r < r else 0)
                dc = 1 if center_c > c else (-1 if center_c < c else 0)
                
                # Mark the line
                max_steps = max(inner_height, inner_width)
                for _ in range(max_steps):
                    # Convert to rectangle coordinates and mark
                    output_rect[r + 1, c + 1] = special_val
                    
                    # Check if we've reached the center (within 0.5 units)
                    if abs(center_r - r) <= 0.5 and abs(center_c - c) <= 0.5:
                        break
                    
                    # Step towards center
                    if abs(center_r - r) > 0.5:
                        r += dr
                    if abs(center_c - c) > 0.5:
                        c += dc
            
            # Update the result grid
            result[min_r:max_r+1, min_c:max_c+1] = output_rect
    
    return result.tolist()


if __name__ == "__main__":
    import json
    
    # Load the task
    task_path = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/8cb8642d.json"
    with open(task_path) as f:
        task = json.load(f)
    
    # Test on all training examples
    all_pass = True
    for idx, example in enumerate(task["train"]):
        input_grid = example["input"]
        expected_output = np.array(example["output"])
        actual_output = np.array(solve(input_grid))
        
        if np.array_equal(expected_output, actual_output):
            print(f"✓ Training example {idx + 1} PASSED")
        else:
            print(f"✗ Training example {idx + 1} FAILED")
            all_pass = False
            
            # Show differences
            diff_mask = expected_output != actual_output
            if diff_mask.any():
                diff_count = np.sum(diff_mask)
                print(f"  {diff_count} cells differ")
                
                # Show first few differences
                diff_coords = np.argwhere(diff_mask)
                for r, c in diff_coords[:10]:
                    print(f"    [{r},{c}]: expected {expected_output[r,c]}, got {actual_output[r,c]}")
    
    if all_pass:
        print("\n✓ All training examples passed!")
    else:
        print("\n✗ Some training examples failed")
