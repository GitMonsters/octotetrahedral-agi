#!/usr/bin/env python3

import json
import copy

def solve(grid):
    """
    ARC-AGI puzzle 6b7e1999 solver - Point symmetry completion around centers
    
    Rule: Find cells with value 7 (when 7 is not background) and complete
    point symmetry (180-degree rotation) around each center within local regions.
    """
    result = copy.deepcopy(grid)
    H, W = len(grid), len(grid[0])
    
    # Find background color (most frequent)
    color_counts = {}
    for r in range(H):
        for c in range(W):
            color = grid[r][c]
            color_counts[color] = color_counts.get(color, 0) + 1
    
    bg_color = max(color_counts, key=color_counts.get)
    
    # Find center points (7s when 7 is not background)
    centers = []
    if bg_color != 7:
        for r in range(H):
            for c in range(W):
                if grid[r][c] == 7:
                    centers.append((r, c))
    
    # For each center, complete point symmetry in surrounding region
    for center_r, center_c in centers:
        # Define region size (try different sizes)
        for radius in range(2, 4):  # 5x5 to 7x7 regions
            top_r = max(0, center_r - radius)
            bottom_r = min(H - 1, center_r + radius)
            left_c = max(0, center_c - radius)
            right_c = min(W - 1, center_c + radius)
            
            # Complete point symmetry in this region
            for r in range(top_r, bottom_r + 1):
                for c in range(left_c, right_c + 1):
                    if result[r][c] != bg_color:
                        # Find point-symmetric position
                        sym_r = 2 * center_r - r
                        sym_c = 2 * center_c - c
                        
                        # If symmetric position is in bounds and empty
                        if (top_r <= sym_r <= bottom_r and 
                            left_c <= sym_c <= right_c and
                            result[sym_r][sym_c] == bg_color):
                            result[sym_r][sym_c] = result[r][c]
    
    return result

def test_solver():
    """Test the solver on training examples"""
    with open('/Users/evanpieser/apr12_tasks/6b7e1999.json', 'r') as f:
        task = json.load(f)
    
    print("Testing point symmetry solver...")
    
    all_correct = True
    for i, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        predicted_output = solve(input_grid)
        
        correct = predicted_output == expected_output
        all_correct = all_correct and correct
        
        print(f"Train {i+1}: {'✓' if correct else '✗'}")
        
        if not correct:
            print("  Differences (first 10):")
            diff_count = 0
            for r in range(len(expected_output)):
                for c in range(len(expected_output[0])):
                    if predicted_output[r][c] != expected_output[r][c]:
                        print(f"    ({r},{c}): expected {expected_output[r][c]}, got {predicted_output[r][c]}")
                        diff_count += 1
                        if diff_count >= 10:
                            break
                if diff_count >= 10:
                    break
    
    print(f"\nOverall: {'All correct!' if all_correct else 'Some failures'}")
    return all_correct

if __name__ == '__main__':
    test_solver()