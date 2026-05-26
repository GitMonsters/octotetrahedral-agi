#!/usr/bin/env python3
"""
Visual examination of the actual transformation by printing grids
"""
import json
import numpy as np

def load_task(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def print_grid(grid, title="Grid"):
    print(f"\n{title}:")
    for row in grid:
        print(''.join(f'{cell:2}' for cell in row))

def print_comparison(input_grid, output_grid, pair_num):
    print(f"\n{'='*50}")
    print(f"TRAIN PAIR {pair_num}")
    print(f"{'='*50}")
    
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)
    
    # Find background color
    unique, counts = np.unique(input_arr, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    
    print(f"Background color: {bg_color}")
    print_grid(input_grid, "INPUT")
    print_grid(output_grid, "OUTPUT")
    
    # Show just the differences
    h, w = input_arr.shape
    diff_grid = []
    for r in range(h):
        row = []
        for c in range(w):
            if input_arr[r, c] != output_arr[r, c]:
                row.append('X')  # Changed
            elif input_arr[r, c] != bg_color:
                row.append('P')  # Pattern
            else:
                row.append('.')  # Background
        diff_grid.append(row)
    
    print("\nCHANGES (X=changed, P=pattern, .=background):")
    for row in diff_grid:
        print(''.join(f'{cell:2}' for cell in row))

def main():
    task = load_task('/Users/evanpieser/apr12_tasks/4e600a86.json')
    
    for i, pair in enumerate(task['train']):
        print_comparison(pair['input'], pair['output'], i + 1)

if __name__ == '__main__':
    main()