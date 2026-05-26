#!/usr/bin/env python3

import json
import copy

def solve(grid):
    """
    ARC-AGI puzzle 6b7e1999 solver - Point symmetry completion
    
    Rule: Find 7s (when not background) and complete point symmetry around them.
    The transformation fills in missing cells to create perfect 180-degree 
    rotational symmetry around each center.
    """
    result = copy.deepcopy(grid)
    H, W = len(grid), len(grid[0])
    
    # Find background color
    color_counts = {}
    for r in range(H):
        for c in range(W):
            color = grid[r][c]
            color_counts[color] = color_counts.get(color, 0) + 1
    bg_color = max(color_counts, key=color_counts.get)
    
    # Find center points
    centers = []
    if bg_color != 7:
        # When 7 is not background, 7s are the centers
        for r in range(H):
            for c in range(W):
                if grid[r][c] == 7:
                    centers.append((r, c))
    else:
        # When 7 is background, find other pattern centers
        # This is more complex - for now focus on non-7 background case
        return grid
    
    def complete_symmetry_around_center(center_r, center_c):
        """Complete point symmetry around a single center"""
        radius = 3
        
        # Multiple passes to ensure convergence
        for iteration in range(5):
            changed = False
            
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    r, c = center_r + dr, center_c + dc
                    
                    if not (0 <= r < H and 0 <= c < W):
                        continue
                        
                    sym_r = 2 * center_r - r
                    sym_c = 2 * center_c - c
                    
                    if not (0 <= sym_r < H and 0 <= sym_c < W):
                        continue
                    
                    val1 = result[r][c]
                    val2 = result[sym_r][sym_c]
                    
                    # Case 1: One is background, other is not -> copy non-background
                    if val1 != bg_color and val2 == bg_color:
                        result[sym_r][sym_c] = val1
                        changed = True
                    elif val2 != bg_color and val1 == bg_color:
                        result[r][c] = val2
                        changed = True
                    # Case 2: Both are background but should be same non-background
                    # Look at symmetric neighbors for clues
                    elif val1 == bg_color and val2 == bg_color:
                        # Check if there's a pattern that suggests both should be something else
                        # Look for nearby non-background values that could extend here
                        for neighbor_dr, neighbor_dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = r + neighbor_dr, c + neighbor_dc
                            if 0 <= nr < H and 0 <= nc < W and result[nr][nc] != bg_color:
                                # Check if the symmetric position of this neighbor is also the same
                                nsym_r = 2 * center_r - nr
                                nsym_c = 2 * center_c - nc  
                                if (0 <= nsym_r < H and 0 <= nsym_c < W and 
                                    result[nsym_r][nsym_c] == result[nr][nc]):
                                    # This suggests a pattern - both positions might need this value
                                    result[r][c] = result[nr][nc]
                                    result[sym_r][sym_c] = result[nr][nc]
                                    changed = True
                                    break
            
            if not changed:
                break
    
    # Apply to each center
    for center_r, center_c in centers:
        complete_symmetry_around_center(center_r, center_c)
    
    return result

def test_final_solver():
    """Test the final solver"""
    with open('/Users/evanpieser/apr12_tasks/6b7e1999.json', 'r') as f:
        task = json.load(f)
    
    print("Testing final solver...")
    
    success_count = 0
    for i, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        predicted_output = solve(input_grid)
        
        correct = predicted_output == expected_output
        if correct:
            success_count += 1
            
        print(f"Train {i+1}: {'✓' if correct else '✗'}")
        
        if not correct:
            # Show just a few differences
            diff_count = 0
            for r in range(len(expected_output)):
                for c in range(len(expected_output[0])):
                    if predicted_output[r][c] != expected_output[r][c]:
                        if diff_count < 3:
                            print(f"    ({r},{c}): expected {expected_output[r][c]}, got {predicted_output[r][c]}")
                        diff_count += 1
            if diff_count > 3:
                print(f"    ... and {diff_count - 3} more differences")
    
    print(f"\nSuccess rate: {success_count}/{len(task['train'])}")
    return success_count == len(task['train'])

if __name__ == '__main__':
    test_final_solver()