#!/usr/bin/env python3
"""
Solver for ARC puzzle 5a5a2103.

The grid is divided by lines (separator color) into rectangular cells.
Each cell that has a non-zero pattern should be replicated to all empty cells,
tile-style, while preserving the separators.
"""

import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solve the ARC puzzle.
    
    The grid is divided by separator lines into cells (rows and columns).
    One cell has a "template" shape (more pixels than a 2x2 block).
    For each row of cells: take the color from the FIRST COLUMN cell,
    apply the template shape with that color to ALL cells in that row.
    """
    if not grid or not grid[0]:
        return grid
    
    height = len(grid)
    width = len(grid[0])
    result = [row[:] for row in grid]  # Start with a copy
    
    # Find separator color
    separator = None
    for val in range(10):
        for row in grid:
            if all(v == val for v in row):
                separator = val
                break
        if separator is not None:
            break
    
    if separator is None:
        return grid
    
    # Find sep rows/cols
    sep_rows = set(i for i in range(height) if all(grid[i][j] == separator for j in range(width)))
    sep_cols = set(j for j in range(width) if all(grid[i][j] == separator for i in range(height)))
    
    # Find row and col groups
    row_groups = []
    col_groups = []
    
    current_group = []
    for i in range(height):
        if i in sep_rows:
            if current_group:
                row_groups.append(current_group)
            current_group = []
        else:
            current_group.append(i)
    if current_group:
        row_groups.append(current_group)
    
    current_group = []
    for j in range(width):
        if j in sep_cols:
            if current_group:
                col_groups.append(current_group)
            current_group = []
        else:
            current_group.append(j)
    if current_group:
        col_groups.append(current_group)
    
    # Count pixels by color (excluding separator and 0)
    color_pixel_count = {}
    for row in grid:
        for val in row:
            if val != separator and val != 0:
                color_pixel_count[val] = color_pixel_count.get(val, 0) + 1
    
    # Find template: color with > 4 pixels (since 2x2 blocks have 4)
    template_color = None
    template_shape = None
    
    if color_pixel_count:
        for color, count in sorted(color_pixel_count.items(), key=lambda x: -x[1]):
            if count > 4:
                template_color = color
                # Find the cell with this color and extract the shape
                for i, row_indices in enumerate(row_groups):
                    for j, col_indices in enumerate(col_groups):
                        cell_data = [[grid[r][c] for c in col_indices] for r in row_indices]
                        has_template = any(cell_data[r][c] == template_color for r in range(len(cell_data)) for c in range(len(cell_data[r])))
                        
                        if has_template:
                            template_shape = [[1 if v == template_color else 0 for v in row] for row in cell_data]
                            break
                    if template_shape:
                        break
                break
    
    # Apply template to all cells
    if template_shape is not None:
        template_height = len(template_shape)
        template_width = len(template_shape[0]) if template_shape else 0
        
        for i, row_indices in enumerate(row_groups):
            # Get the color for this row: color of the first cell in this row
            first_cell_color = None
            if col_groups:
                col_indices = col_groups[0]
                cell_data = [[grid[r][c] for c in col_indices] for r in row_indices]
                for r in range(len(cell_data)):
                    for c in range(len(cell_data[r])):
                        if cell_data[r][c] != separator and cell_data[r][c] != 0:
                            first_cell_color = cell_data[r][c]
                            break
                    if first_cell_color:
                        break
            
            # Apply template shape with first_cell_color to all cells in this row
            if first_cell_color:
                for j, col_indices in enumerate(col_groups):
                    for r_idx, r in enumerate(row_indices):
                        for c_idx, c in enumerate(col_indices):
                            if r_idx < template_height and c_idx < template_width:
                                if template_shape[r_idx][c_idx] == 1:
                                    result[r][c] = first_cell_color
                                else:
                                    result[r][c] = 0
    
    return result


def main():
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/5a5a2103.json") as f:
        data = json.load(f)
    
    # Test on training examples
    passed = 0
    failed = 0
    for idx, example in enumerate(data["train"]):
        output = solve(example["input"])
        expected = example["output"]
        
        if output == expected:
            print(f"PASS: training example {idx}")
            passed += 1
        else:
            print(f"FAIL: training example {idx}")
            failed += 1
            # Print differences for debugging
            for r in range(len(output)):
                for c in range(len(output[r])):
                    if output[r][c] != expected[r][c]:
                        print(f"  Mismatch at ({r}, {c}): got {output[r][c]}, expected {expected[r][c]}")
                        break
                else:
                    continue
                break
    
    print(f"\nResults: {passed} passed, {failed} failed")


if __name__ == "__main__":
    main()
