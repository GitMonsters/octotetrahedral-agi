import json
import sys
import numpy as np
from collections import defaultdict, Counter

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solve ARC-AGI task 4b6b68e5.
    
    Rule: For each rectangular region bounded by a colored border:
    1. Find all non-zero, non-border values inside the rectangle
    2. Determine which value appears most frequently (or only appears)
    3. Fill the entire interior of the rectangle with that value
    """
    
    grid = [row[:] for row in grid]  # Deep copy
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    if height == 0 or width == 0:
        return grid
    
    # Convert to numpy for easier manipulation
    arr = np.array(grid)
    
    # Find all potential border colors (excluding 0 and 8, which are special)
    # 8 is special marker, 0 is background
    potential_borders = set()
    for row in grid:
        for val in row:
            if val > 0 and val not in [8]:
                potential_borders.add(val)
    
    # For each potential border color, try to find and fill rectangles
    for border_color in sorted(potential_borders):
        # Find all cells with this border color
        border_cells = []
        for i in range(height):
            for j in range(width):
                if grid[i][j] == border_color:
                    border_cells.append((i, j))
        
        if len(border_cells) < 4:
            continue
        
        border_cells = np.array(border_cells)
        min_r = border_cells[:, 0].min()
        max_r = border_cells[:, 0].max()
        min_c = border_cells[:, 1].min()
        max_c = border_cells[:, 1].max()
        
        # Check if this forms a valid rectangle border
        # A valid border should have cells on the perimeter
        is_valid = False
        for r in range(min_r, max_r + 1):
            if grid[r][min_c] == border_color or grid[r][max_c] == border_color:
                is_valid = True
                break
        
        if not is_valid:
            continue
        
        # Find all non-zero, non-border values inside the rectangle
        interior_values = []
        for i in range(min_r + 1, max_r):
            for j in range(min_c + 1, max_c):
                if grid[i][j] != 0 and grid[i][j] != border_color:
                    interior_values.append(grid[i][j])
        
        if not interior_values:
            continue
        
        # Determine the fill color: the most frequent interior value
        value_counts = Counter(interior_values)
        fill_color = value_counts.most_common(1)[0][0]
        
        # Fill the interior of the rectangle (excluding the border itself)
        # We need to be careful to preserve the border and only fill 0s and interior markers
        for i in range(min_r + 1, max_r):
            for j in range(min_c + 1, max_c):
                # Only fill if it's a 0 or one of the interior values that will be replaced
                if grid[i][j] != border_color:
                    grid[i][j] = fill_color
    
    return grid


def main():
    if len(sys.argv) > 1:
        task_path = sys.argv[1]
    else:
        task_path = '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/4b6b68e5.json'
    
    with open(task_path, 'r') as f:
        task_data = json.load(f)
    
    # Test on training examples
    all_pass = True
    for idx, example in enumerate(task_data['train']):
        input_grid = example['input']
        expected_output = example['output']
        
        predicted_output = solve(input_grid)
        
        if predicted_output == expected_output:
            print(f"Training example {idx}: PASS")
        else:
            print(f"Training example {idx}: FAIL")
            all_pass = False
            # Show first difference
            for i in range(len(expected_output)):
                for j in range(len(expected_output[i])):
                    if predicted_output[i][j] != expected_output[i][j]:
                        print(f"  First diff at ({i}, {j}): got {predicted_output[i][j]}, expected {expected_output[i][j]}")
                        break
                else:
                    continue
                break
    
    if all_pass:
        print("\nAll training examples passed!")
    else:
        print("\nSome training examples failed.")
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
