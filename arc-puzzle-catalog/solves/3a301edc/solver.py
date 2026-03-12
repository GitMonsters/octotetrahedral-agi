import json
import sys


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Rule: Find a rectangular region with nested colors (outer and inner).
    
    The transformation expands the rectangle and fills the expanded border region
    with the innermost color, while preserving the original structure in the core.
    
    The expansion amount is: min(margin_to_inner, max(inner_height, inner_width))
    where margin is the distance from the outer rectangle boundary to the inner color.
    """
    grid = [row[:] for row in grid]
    
    # Find bounding box of non-zero elements (outer rectangle)
    rows_with_color = [r for r in range(len(grid)) if any(v != 0 for v in grid[r])]
    cols_with_color = [c for c in range(len(grid[0])) if any(grid[r][c] != 0 for r in range(len(grid)))]
    
    if not rows_with_color or not cols_with_color:
        return grid
    
    outer_min_row, outer_max_row = min(rows_with_color), max(rows_with_color)
    outer_min_col, outer_max_col = min(cols_with_color), max(cols_with_color)
    
    # Find all colors in the outer rectangle
    colors = {}
    for r in range(outer_min_row, outer_max_row + 1):
        for c in range(outer_min_col, outer_max_col + 1):
            v = grid[r][c]
            if v != 0:
                colors[v] = colors.get(v, 0) + 1
    
    if len(colors) < 2:
        return grid
    
    # The rarest color is the innermost one
    sorted_colors = sorted(colors.items(), key=lambda x: x[1])
    inner_color = sorted_colors[0][0]
    
    # Find bounding box of inner color
    inner_positions = [(r, c) for r in range(outer_min_row, outer_max_row + 1)
                       for c in range(outer_min_col, outer_max_col + 1) if grid[r][c] == inner_color]
    
    if not inner_positions:
        return grid
    
    inner_rows = [r for r, c in inner_positions]
    inner_cols = [c for r, c in inner_positions]
    inner_min_row, inner_max_row = min(inner_rows), max(inner_rows)
    inner_min_col, inner_max_col = min(inner_cols), max(inner_cols)
    
    # Calculate dimensions and margin
    inner_height = inner_max_row - inner_min_row + 1
    inner_width = inner_max_col - inner_min_col + 1
    margin = inner_min_row - outer_min_row
    
    # Expansion amount is the minimum of the margin and the maximum inner dimension
    expansion = min(margin, max(inner_height, inner_width))
    
    # Save original content
    orig_content = {}
    for r in range(outer_min_row, outer_max_row + 1):
        for c in range(outer_min_col, outer_max_col + 1):
            orig_content[(r, c)] = grid[r][c]
    
    # Calculate new boundaries
    new_min_row = max(0, outer_min_row - expansion)
    new_max_row = min(len(grid) - 1, outer_max_row + expansion)
    new_min_col = max(0, outer_min_col - expansion)
    new_max_col = min(len(grid[0]) - 1, outer_max_col + expansion)
    
    # Fill the entire expanded region with inner_color
    for r in range(new_min_row, new_max_row + 1):
        for c in range(new_min_col, new_max_col + 1):
            grid[r][c] = inner_color
    
    # Restore the original content in the original rectangle
    for (r, c), v in orig_content.items():
        grid[r][c] = v
    
    return grid


if __name__ == "__main__":
    # Load task JSON
    if len(sys.argv) > 1:
        task_path = sys.argv[1]
    else:
        task_path = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/3a301edc.json"
    
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    # Test on all training examples
    all_pass = True
    for i, pair in enumerate(task['train']):
        result = solve(pair['input'])
        expected = pair['output']
        
        if result == expected:
            print(f"Training example {i}: PASS")
        else:
            print(f"Training example {i}: FAIL")
            all_pass = False
            # Show first few differences
            diffs = 0
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        diffs += 1
                        if diffs <= 3:
                            print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
            if diffs > 3:
                print(f"  ... and {diffs - 3} more differences")
    
    print(f"\nAll training examples pass: {all_pass}")
