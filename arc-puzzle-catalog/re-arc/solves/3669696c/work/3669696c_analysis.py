#!/usr/bin/env python3

import json

# Load task data
with open('/Users/evanpieser/apr12_tasks/3669696c.json', 'r') as f:
    data = json.load(f)

def print_grid(grid, label=""):
    if label:
        print(f"\n{label}:")
    for row in grid:
        print(''.join(str(x) for x in row))

def analyze_example(idx):
    train_example = data['train'][idx]
    input_grid = train_example['input']
    expected_output = train_example['output']
    
    print(f"=== TRAINING EXAMPLE {idx} ===")
    print_grid(input_grid, "INPUT")
    print_grid(expected_output, "EXPECTED OUTPUT")
    
    # Find all gray (5) and black (0) positions
    height = len(input_grid)
    width = len(input_grid[0])
    
    colored_positions = []
    for r in range(height):
        for c in range(width):
            if input_grid[r][c] in [0, 5]:
                colored_positions.append((r, c, input_grid[r][c]))
                print(f"Found color {input_grid[r][c]} at position ({r}, {c})")
    
    # Check if diagonal rays are drawn from these positions
    print("\nChecking diagonal rays...")
    for r, c, color in colored_positions:
        # Check 4 diagonal directions: NE, NW, SE, SW
        directions = [(-1, 1), (-1, -1), (1, 1), (1, -1)]  # (dr, dc)
        
        for dr, dc in directions:
            # Walk along this diagonal
            nr, nc = r + dr, c + dc
            ray_positions = []
            while 0 <= nr < height and 0 <= nc < width:
                ray_positions.append((nr, nc))
                nr += dr
                nc += dc
            
            if ray_positions:
                print(f"  From ({r},{c}) direction ({dr},{dc}): ray hits {ray_positions[:3]}...")
                # Check if these positions have the color in expected output
                for rr, cc in ray_positions[:3]:  # Check first few
                    if expected_output[rr][cc] == color:
                        print(f"    ✓ ({rr},{cc}) has color {color} in output")
                    else:
                        print(f"    ✗ ({rr},{cc}) has color {expected_output[rr][cc]} in output, expected {color}")

# Analyze first example
analyze_example(0)