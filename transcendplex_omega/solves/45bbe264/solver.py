import json
import sys


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Find all non-zero values and their positions.
    For each value at (r, c):
    - Draw a vertical line in column c with that value
    - Draw a horizontal line in row r with that value
    - Where they intersect, place the value 2
    """
    height = len(grid)
    width = len(grid[0])
    
    # Create output grid (copy of input initially)
    output = [row[:] for row in grid]
    
    # Find all non-zero positions
    marks = []
    for r in range(height):
        for c in range(width):
            if grid[r][c] != 0:
                marks.append((r, c, grid[r][c]))
    
    # For each mark, draw lines
    for mark_r, mark_c, color in marks:
        # Draw vertical line in column mark_c
        for r in range(height):
            if output[r][mark_c] == 0:
                output[r][mark_c] = color
        
        # Draw horizontal line in row mark_r
        for c in range(width):
            if output[mark_r][c] == 0:
                output[mark_r][c] = color
    
    # Place 2s at intersections (where two different lines cross)
    for mark_r, mark_c, color in marks:
        # Check intersections with other marks
        for other_r, other_c, other_color in marks:
            if (mark_r, mark_c) != (other_r, other_c):
                # Intersection at (mark_r, other_c) and (other_r, mark_c)
                output[mark_r][other_c] = 2
                output[other_r][mark_c] = 2
    
    return output


if __name__ == "__main__":
    # Load task JSON
    task_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/45bbe264.json"
    
    with open(task_path) as f:
        task = json.load(f)
    
    # Test on all training examples
    all_pass = True
    for idx, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        actual_output = solve(input_grid)
        
        if actual_output == expected_output:
            print(f"PASS: Training example {idx + 1}")
        else:
            print(f"FAIL: Training example {idx + 1}")
            all_pass = False
            print(f"Expected:\n{expected_output}")
            print(f"Got:\n{actual_output}")
    
    if all_pass:
        print("\n✓ All training examples PASS")
    else:
        print("\n✗ Some training examples FAILED")
        sys.exit(1)
