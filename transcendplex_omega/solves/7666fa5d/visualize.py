#!/usr/bin/env python3

import json

def visualize_example(input_grid, output_grid, example_num):
    """Create a side-by-side visualization of input and output"""
    rows, cols = len(input_grid), len(input_grid[0])
    
    print(f"\n=== EXAMPLE {example_num} ===")
    print("INPUT" + " " * (cols-2) + "OUTPUT")
    
    for r in range(rows):
        # Input row
        input_row = ""
        for c in range(cols):
            val = input_grid[r][c]
            if val == 8:
                input_row += "."
            else:
                input_row += str(val)
        
        # Output row
        output_row = ""
        for c in range(cols):
            val = output_grid[r][c]
            if val == 8:
                output_row += "."
            elif val == 2:
                output_row += "#"
            else:
                output_row += str(val)
        
        print(f"{input_row}  {output_row}")
    
    # Show just the changes
    print("\nCHANGES (positions where 2's were added):")
    changes = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != output_grid[r][c]:
                changes.append((r, c))
    
    # Group changes by row for easier pattern recognition
    by_row = {}
    for r, c in changes:
        if r not in by_row:
            by_row[r] = []
        by_row[r].append(c)
    
    for r in sorted(by_row.keys()):
        cols_changed = sorted(by_row[r])
        print(f"Row {r:2d}: cols {cols_changed}")

def main():
    with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/7666fa5d.json', 'r') as f:
        data = json.load(f)
    
    for i, example in enumerate(data['train']):
        visualize_example(example['input'], example['output'], i+1)

if __name__ == "__main__":
    main()