import json
import sys
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    The puzzle:
    1. Top section (rows 0-4) has 3 shape patterns with different colors
    2. Middle row (5) is a separator (all 5s)
    3. Bottom section (rows 6+) has a single shape pattern
    4. Output: vertically stack shapes that are NOT at the bottom shape's column position
    
    The rule:
    - Find column position of bottom shape
    - Find top shapes that are NOT at the same column position
    - Stack them vertically at the bottom shape's column, centered
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
    
    # Find top shapes NOT at bottom column and extract them
    shapes_to_stack = []
    for color in sorted(top_shapes.keys(), reverse=False):
        bbox = get_bbox(top_shapes[color])
        col_min, col_max = bbox[2], bbox[3]
        col_center = (col_min + col_max) // 2
        
        # Check if this color is NOT at the same column position as bottom
        if col_center != bottom_col_center:
            pattern_data = extract_pattern(top_shapes[color])
            if pattern_data:
                pattern, _ = pattern_data
                shapes_to_stack.append((color, pattern))
    
    # Extract bottom shape pattern
    bottom_pattern_data = extract_pattern(bottom_shapes[bottom_color])
    bottom_pattern, _ = bottom_pattern_data
    
    # Build output: calculate total height and width needed
    total_height = sum(len(p[1]) for p in shapes_to_stack) + len(bottom_pattern) + 1  # +1 for spacing
    output_width = len(grid[0])
    output = [[0] * output_width for _ in range(total_height)]
    
    # Place shapes vertically, centered at bottom_col_center
    out_row = 0
    all_shapes = shapes_to_stack + [(bottom_color, bottom_pattern)]
    
    for color, pattern in all_shapes:
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
        
        out_row += pattern_height + 1  # +1 for spacing between shapes
    
    # Trim output - remove trailing empty rows
    while output and all(cell == 0 for cell in output[-1]):
        output.pop()
    
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
