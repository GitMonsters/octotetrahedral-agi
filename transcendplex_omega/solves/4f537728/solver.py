import json
import sys


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Find the special color (non-0, non-1 value) in the grid.
    Paint the entire column(s) and row(s) containing that color with it.
    """
    # Create output as a copy
    output = [row[:] for row in grid]
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    # Find all non-zero, non-one values (the special color)
    special_color = None
    special_positions = []
    
    for r in range(height):
        for c in range(width):
            val = grid[r][c]
            if val not in (0, 1):
                if special_color is None:
                    special_color = val
                special_positions.append((r, c))
    
    if special_color is None:
        return output
    
    # Find which rows and columns contain the special color
    special_rows = set(r for r, c in special_positions)
    special_cols = set(c for r, c in special_positions)
    
    # Paint entire columns and rows with the special color
    for r in range(height):
        for c in range(width):
            # Skip rows and columns that are all zeros
            if grid[r][c] != 0:
                if r in special_rows or c in special_cols:
                    output[r][c] = special_color
    
    return output


if __name__ == "__main__":
    # Load task JSON (default path or from argument)
    task_path = sys.argv[1] if len(sys.argv) > 1 else \
        "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/4f537728.json"
    task_path = task_path.replace("~", "/Users/evanpieser")
    
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    # Test on training examples
    all_pass = True
    for idx, example in enumerate(task['train']):
        predicted = solve(example['input'])
        expected = example['output']
        
        if predicted == expected:
            print(f"Training example {idx}: PASS")
        else:
            print(f"Training example {idx}: FAIL")
            all_pass = False
            # Show difference details
            for r in range(len(predicted)):
                for c in range(len(predicted[r])):
                    if predicted[r][c] != expected[r][c]:
                        print(f"  Mismatch at ({r},{c}): got {predicted[r][c]}, expected {expected[r][c]}")
                        break
                if not all_pass:
                    break
    
    if all_pass:
        print("\nAll training examples PASSED!")
    else:
        print("\nSome training examples FAILED!")
        sys.exit(1)
