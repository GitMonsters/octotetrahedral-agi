import json
import numpy as np

def analyze_exact_patterns():
    with open('/Users/evanpieser/apr12_tasks/12191b16.json', 'r') as f:
        task = json.load(f)
    
    print("=== EXACT PATTERN ANALYSIS ===")
    
    for pair_idx, pair in enumerate(task['train']):
        print(f"\n--- PAIR {pair_idx + 1} ---")
        
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        # Find background
        unique, counts = np.unique(input_grid, return_counts=True)
        bg = unique[np.argmax(counts)]
        
        print(f"Background: {bg}")
        
        # Find dots
        dots = []
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                if input_grid[i,j] != bg:
                    dots.append((i, j, input_grid[i,j]))
        
        print("Input dots:", dots)
        
        # Analyze specific output rows
        h, w = output_grid.shape
        
        print(f"Output size: {h}x{w}")
        
        # Check specific rows that have patterns
        for row in range(min(10, h)):
            output_row = output_grid[row]
            non_bg_positions = [i for i, val in enumerate(output_row) if val != bg]
            
            if non_bg_positions:
                print(f"Row {row}: positions {non_bg_positions}")
                print(f"  Values: {[output_row[i] for i in non_bg_positions]}")
                
                # Check if there's a simple pattern
                if len(non_bg_positions) > 2:
                    # Check for symmetry
                    left_half = non_bg_positions[:len(non_bg_positions)//2]
                    right_half = non_bg_positions[len(non_bg_positions)//2:]
                    right_half_mirrored = [w-1-pos for pos in reversed(right_half)]
                    
                    print(f"  Left positions: {left_half}")  
                    print(f"  Right positions mirrored: {right_half_mirrored}")
                    
                    if left_half == right_half_mirrored:
                        print("  -> SYMMETRIC!")

analyze_exact_patterns()