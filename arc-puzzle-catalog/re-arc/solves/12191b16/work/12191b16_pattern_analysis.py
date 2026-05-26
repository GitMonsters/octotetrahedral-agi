import json
import numpy as np

def detailed_pattern_analysis(pair_num):
    with open('/Users/evanpieser/apr12_tasks/12191b16.json', 'r') as f:
        task = json.load(f)
    
    pair = task['train'][pair_num - 1]
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    print(f"\n=== DETAILED ANALYSIS PAIR {pair_num} ===")
    
    # Find background
    unique, counts = np.unique(input_grid, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    
    # Find all non-background positions
    dots = []
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            if input_grid[i, j] != bg_color:
                dots.append((i, j, input_grid[i, j]))
    
    print("Input dots:", dots)
    
    # Check what happens in the output at these positions
    print("Output at dot positions:")
    for i, j, color in dots:
        print(f"  ({i},{j}) input:{color} -> output:{output_grid[i,j]}")
    
    # Find bounding box
    rows = [d[0] for d in dots]
    cols = [d[1] for d in dots]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    print(f"Bounding box: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
    
    # Analyze the symmetric extension
    print(f"\nOutput grid center: row {output_grid.shape[0]//2}, col {output_grid.shape[1]//2}")
    
    # Check if the pattern is symmetric
    center_row = output_grid.shape[0] // 2
    center_col = output_grid.shape[1] // 2
    
    print(f"Looking at symmetry around center ({center_row}, {center_col})")
    
    # Check a few key rows for patterns
    for row_idx in [1, 3, 5, 7, 9]:
        if row_idx < output_grid.shape[0]:
            row = output_grid[row_idx]
            print(f"Row {row_idx}: {row[:15]}...")  # First 15 elements
    
    return dots, min_row, max_row, min_col, max_col

# Analyze all pairs
for i in range(1, 4):
    detailed_pattern_analysis(i)