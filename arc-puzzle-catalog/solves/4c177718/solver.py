import json
import sys
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    ARC-AGI Task 4c177718: Shape Reordering and Stacking
    
    Input structure:
    - Rows 0-4: Top section with 3 colored shapes (left, middle, right columns)
    - Row 5: Separator row (all 5s)
    - Rows 6+: Bottom section with 1 colored shape (matching color from top-left)
    
    Transformation rule:
    1. Skip the middle shape from the top section
    2. Take the left and right shapes from the top
    3. Replace the selected shape's pattern with the bottom shape's pattern where colors match
    4. Stack the shapes vertically at the bottom shape's column position
    5. Output order depends on distance: if bottom closer to left, order is [right, left];
       if bottom closer to right or equal, order is [left, right]
    6. Vertical positioning: if left-color-first, starts at offset=bot_first_offset;
       if right-color-first, starts at offset=max(0, bot_first_offset-3)
    7. Output is always 9 rows × 15 cols, with shapes centered at the bottom shape's column
    """
    
    # Find separator row (all 5s)
    sep_row = -1
    for i, row in enumerate(grid):
        if all(cell == 5 for cell in row):
            sep_row = i
            break
    
    # Split grid
    top_section = grid[:sep_row]
    bottom_section = grid[sep_row + 1:]
    
    # Find all colored shapes
    def get_shapes(section):
        shapes = {}  # color -> list of (row, col) positions
        for r in range(len(section)):
            for c in range(len(section[r])):
                val = section[r][c]
                if val not in [0, 5]:
                    if val not in shapes:
                        shapes[val] = []
                    shapes[val].append((r, c))
        return shapes
    
    top_shapes = get_shapes(top_section)
    bottom_shapes = get_shapes(bottom_section)
    
    def get_bbox(positions):
        if not positions:
            return None
        min_r = min(p[0] for p in positions)
        max_r = max(p[0] for p in positions)
        min_c = min(p[1] for p in positions)
        max_c = max(p[1] for p in positions)
        return (min_r, max_r, min_c, max_c)
    
    def extract_pattern(positions):
        """Extract shape pattern (1s and 0s) relative to bounding box"""
        bbox = get_bbox(positions)
        if not bbox:
            return None
        min_r, max_r, min_c, max_c = bbox
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        pattern = [[0] * width for _ in range(height)]
        for r, c in positions:
            pattern[r - min_r][c - min_c] = 1
        return pattern, (min_r, min_c)  # return pattern and top-left coord
    
    # Get bottom shape column position
    bottom_color = list(bottom_shapes.keys())[0]
    bottom_bbox = get_bbox(bottom_shapes[bottom_color])
    bottom_col_min, bottom_col_max = bottom_bbox[2], bottom_bbox[3]
    bottom_col_center = (bottom_col_min + bottom_col_max) // 2
    
    # Find top shapes: we want LEFT and RIGHT, skip MIDDLE
    # Sort colors by their column position
    color_positions = []
    for color in top_shapes:
        bbox = get_bbox(top_shapes[color])
        col_min, col_max = bbox[2], bbox[3]
        col_center = (col_min + col_max) // 2
        color_positions.append((col_center, color))
    
    color_positions.sort()
    left_color = color_positions[0][1]
    middle_color = color_positions[1][1]  # This is always skipped
    right_color = color_positions[2][1]
    
    # Determine output order based on bottom shape's position
    dist_to_left = abs(bottom_col_center - color_positions[0][0])
    dist_to_right = abs(bottom_col_center - color_positions[2][0])
    
    if dist_to_left < dist_to_right:
        # Bottom is closer to left → output order is [right, left]
        output_colors = [right_color, left_color]
    else:
        # Bottom is closer to right or equal → output order is [left, right]
        output_colors = [left_color, right_color]
    
    # Collect shapes in output order
    # If a shape color matches the bottom color, use the bottom's pattern instead
    shapes_to_stack = []
    for color in output_colors:
        if color == bottom_color:
            # Use bottom pattern
            bottom_pattern_data = extract_pattern(bottom_shapes[bottom_color])
            bottom_pattern, _ = bottom_pattern_data
            shapes_to_stack.append((color, bottom_pattern))
        else:
            # Use top pattern
            pattern_data = extract_pattern(top_shapes[color])
            if pattern_data:
                pattern, _ = pattern_data
                shapes_to_stack.append((color, pattern))
    
    # Determine starting row in output based on bottom shape's position
    # Find first row of bottom shape (after separator)
    bot_first_row = -1
    for r in range(sep_row + 1, len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] != 0:
                if bot_first_row == -1:
                    bot_first_row = r
                break
    
    bot_first_offset = bot_first_row - (sep_row + 1)  # Offset from row (sep_row + 1)
    
    # Calculate output starting row based on output order and bottom offset
    if output_colors[0] == left_color:
        # Left color first: output_start = bot_first_offset
        output_start_row = bot_first_offset
    else:
        # Right color first: output_start = max(0, bot_first_offset - 3)
        output_start_row = max(0, bot_first_offset - 3)
    
    # Build output with the correct size
    # All outputs are 9 rows
    output_width = len(grid[0])
    output = [[0] * output_width for _ in range(9)]
    
    # Place shapes starting at output_start_row
    out_row = output_start_row
    
    for color, pattern in shapes_to_stack:
        pattern_height = len(pattern)
        pattern_width = len(pattern[0]) if pattern else 0
        
        # Center the pattern horizontally at bottom_col_center
        col_start = bottom_col_center - pattern_width // 2
        
        # Place pattern
        for pr in range(pattern_height):
            for pc in range(pattern_width):
                if pattern[pr][pc] == 1:
                    if 0 <= col_start + pc < output_width and out_row + pr < len(output):
                        output[out_row + pr][col_start + pc] = color
        
        out_row += pattern_height  # No spacing between shapes
    
    # Return the output grid (always 9 rows)
    return output


def main():
    if len(sys.argv) > 1:
        task_path = sys.argv[1]
    else:
        task_path = os.path.expanduser("~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/4c177718.json")
    
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    # Test on training examples
    all_pass = True
    for idx, example in enumerate(task['train']):
        result = solve(example['input'])
        expected = example['output']
        
        if result == expected:
            print(f"Training example {idx}: PASS")
        else:
            print(f"Training example {idx}: FAIL")
            print(f"  Expected: {len(expected)} rows, {len(expected[0]) if expected else 0} cols")
            print(f"  Got:      {len(result)} rows, {len(result[0]) if result else 0} cols")
            all_pass = False
    
    if all_pass:
        print("\nAll training examples PASSED!")
    else:
        print("\nSome examples failed.")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    import os
    sys.exit(main())
