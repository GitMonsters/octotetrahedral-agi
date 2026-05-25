import json
import sys
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    Extract the pattern formed by 8s and tile it horizontally based on the count of 4s.
    The number of 4s determines how many times to repeat the 8 pattern horizontally.
    The output is always 3 rows tall (pad with zeros on top if needed).
    """
    
    # Count the number of 4s
    num_tiles = sum(row.count(4) for row in grid)
    
    # Find bounding box of 8s
    rows_with_8 = [i for i in range(len(grid)) if 8 in grid[i]]
    cols_with_8 = [j for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j] == 8]
    
    if not rows_with_8 or not cols_with_8:
        # No 8s found, return empty grid
        return [[]]
    
    min_row = min(rows_with_8)
    max_row = max(rows_with_8)
    min_col = min(cols_with_8)
    max_col = max(cols_with_8)
    
    pattern_height = max_row - min_row + 1
    pattern_width = max_col - min_col + 1
    
    # Extract the pattern
    pattern = []
    for i in range(min_row, max_row + 1):
        row = []
        for j in range(min_col, max_col + 1):
            row.append(grid[i][j])
        pattern.append(row)
    
    # Ensure output is 3 rows tall
    target_height = 3
    if pattern_height < target_height:
        # Pad with zeros on top
        padding = target_height - pattern_height
        pattern = [[0] * pattern_width] * padding + pattern
    elif pattern_height > target_height:
        # Crop to first 3 rows
        pattern = pattern[:target_height]
    
    # Tile the pattern horizontally num_tiles times
    result = []
    for pattern_row in pattern:
        result_row = []
        for _ in range(num_tiles):
            result_row.extend(pattern_row)
        result.append(result_row)
    
    return result


if __name__ == "__main__":
    # Load task JSON
    task_path = sys.argv[1] if len(sys.argv) > 1 else "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/4852f2fa.json"
    task_path = task_path.replace("~", "/Users/evanpieser")
    
    with open(task_path) as f:
        task = json.load(f)
    
    # Test on training examples
    print(f"Testing on {len(task['train'])} training examples:")
    all_pass = True
    
    for idx, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        actual_output = solve(input_grid)
        
        if actual_output == expected_output:
            print(f"  Training {idx}: PASS")
        else:
            print(f"  Training {idx}: FAIL")
            print(f"    Expected: {expected_output}")
            print(f"    Got:      {actual_output}")
            all_pass = False
    
    if all_pass:
        print("\nAll training examples PASS!")
    else:
        print("\nSome training examples FAILED!")
        sys.exit(1)
