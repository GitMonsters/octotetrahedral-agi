#!/usr/bin/env python3
import json
import sys
from copy import deepcopy


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Rule: For each color that appears in multiple rectangular boxes,
    find the box with a non-zero interior pattern and copy that pattern
    to all other boxes of the same color that have hollow interiors.
    """
    result = deepcopy(grid)
    
    # Find all colored rectangular boxes using connected components
    def find_boxes_by_color():
        visited = set()
        boxes_by_color = {}
        
        def dfs(i, j, color, positions):
            if (i, j) in visited or i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
                return
            if grid[i][j] != color:
                return
            
            visited.add((i, j))
            positions.append((i, j))
            
            # Check 4 neighbors
            dfs(i+1, j, color, positions)
            dfs(i-1, j, color, positions)
            dfs(i, j+1, color, positions)
            dfs(i, j-1, color, positions)
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] != 0 and (i, j) not in visited:
                    color = grid[i][j]
                    positions = []
                    dfs(i, j, color, positions)
                    
                    if positions:
                        r_min = min(r for r, c in positions)
                        r_max = max(r for r, c in positions)
                        c_min = min(c for r, c in positions)
                        c_max = max(c for r, c in positions)
                        
                        if color not in boxes_by_color:
                            boxes_by_color[color] = []
                        boxes_by_color[color].append((r_min, r_max, c_min, c_max))
        
        return boxes_by_color
    
    def get_interior(grid, r_min, r_max, c_min, c_max):
        """Extract interior of a box (excluding borders)"""
        interior = []
        for i in range(r_min + 1, r_max):
            row = []
            for j in range(c_min + 1, c_max):
                row.append(grid[i][j])
            interior.append(row)
        return interior
    
    def has_pattern(interior, color):
        """Check if interior has any cells of the color"""
        if not interior or not interior[0]:
            return False
        count = sum(1 for row in interior for cell in row if cell == color)
        return count > 0
    
    boxes_by_color = find_boxes_by_color()
    
    # Filter out boxes that are too small to have an interior pattern
    for color in list(boxes_by_color.keys()):
        boxes_by_color[color] = [
            box for box in boxes_by_color[color]
            if (box[1] - box[0] >= 3 and box[3] - box[2] >= 3)  # At least 3x3
        ]
        if not boxes_by_color[color]:
            del boxes_by_color[color]
    
    # Process colors that appear in multiple boxes
    for color, boxes in boxes_by_color.items():
        if len(boxes) > 1:
            # Find the box with a pattern in its interior
            pattern_box = None
            pattern_interior = None
            
            for r_min, r_max, c_min, c_max in boxes:
                interior = get_interior(grid, r_min, r_max, c_min, c_max)
                if has_pattern(interior, color):
                    pattern_box = (r_min, r_max, c_min, c_max)
                    pattern_interior = interior
                    break
            
            # If we found a pattern, copy it to hollow boxes
            if pattern_interior is not None:
                for r_min, r_max, c_min, c_max in boxes:
                    interior = get_interior(result, r_min, r_max, c_min, c_max)
                    # Check if this is a hollow box (needs filling)
                    if not has_pattern(interior, color):
                        # Copy the pattern
                        for i, row in enumerate(pattern_interior):
                            for j, val in enumerate(row):
                                result[r_min + 1 + i][c_min + 1 + j] = val
    
    return result


def main():
    # Load task JSON
    if len(sys.argv) > 1:
        task_path = sys.argv[1]
    else:
        task_path = '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/42918530.json'
    
    with open(task_path) as f:
        task = json.load(f)
    
    # Test on training examples
    all_pass = True
    for idx, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        
        predicted_output = solve(input_grid)
        
        passed = predicted_output == expected_output
        all_pass = all_pass and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"Training example {idx + 1}: {status}")
        
        if not passed:
            print(f"  Expected output shape: {len(expected_output)} x {len(expected_output[0])}")
            print(f"  Got output shape: {len(predicted_output)} x {len(predicted_output[0])}")
            # Show first difference
            for i in range(len(expected_output)):
                for j in range(len(expected_output[0])):
                    if expected_output[i][j] != predicted_output[i][j]:
                        print(f"  First diff at ({i},{j}): expected {expected_output[i][j]}, got {predicted_output[i][j]}")
                        break
                else:
                    continue
                break
    
    if all_pass:
        print("\nAll training examples PASS")
    else:
        print("\nSome training examples FAILED")
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
