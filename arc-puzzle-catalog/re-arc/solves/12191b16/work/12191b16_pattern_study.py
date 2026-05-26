import json
import numpy as np

# Let me study the exact expected output patterns to understand the rule
with open('/Users/evanpieser/apr12_tasks/12191b16.json', 'r') as f:
    task = json.load(f)

print("=== REVERSE ENGINEERING THE EXACT PATTERN ===")

for pair_idx, pair in enumerate(task['train']):
    print(f"\n--- PAIR {pair_idx + 1} ---")
    
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    # Find background
    unique, counts = np.unique(input_grid, return_counts=True)
    bg = unique[np.argmax(counts)]
    
    # Find input dots
    input_dots = []
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            if input_grid[i,j] != bg:
                input_dots.append((i, j, input_grid[i,j]))
    
    print("Input dots:", input_dots)
    
    # Analyze each output row that has patterns
    h, w = output_grid.shape
    
    # Look at specific key rows  
    for row_idx in [1, 3, 5, 7]:
        if row_idx < h:
            row = output_grid[row_idx]
            print(f"\nRow {row_idx}:")
            print("  Full row:", list(row))
            
            # Find colored positions
            colored_pos = [(j, row[j]) for j in range(w) if row[j] != bg]
            print("  Colored positions:", colored_pos)
            
            # Check if it matches input dots in this row
            input_row_dots = [(j, color) for i, j, color in input_dots if i == row_idx]
            print("  Input dots in this row:", input_row_dots)
            
            # Check for pattern
            if len(colored_pos) > 2:
                # Are the positions regular?
                positions = [pos for pos, _ in colored_pos]
                if len(positions) >= 2:
                    spacing = positions[1] - positions[0]
                    regular = all(positions[i+1] - positions[i] == spacing for i in range(len(positions)-1))
                    print(f"  Regular spacing of {spacing}: {regular}")
                
                # Check symmetry
                left_half = colored_pos[:len(colored_pos)//2]
                right_half = colored_pos[len(colored_pos)//2:]
                symmetric = True
                for i, (left_pos, left_color) in enumerate(left_half):
                    if i < len(right_half):
                        right_pos, right_color = right_half[-(i+1)]
                        expected_right_pos = w - 1 - left_pos
                        if right_pos != expected_right_pos or left_color != right_color:
                            symmetric = False
                            break
                print(f"  Symmetric: {symmetric}")