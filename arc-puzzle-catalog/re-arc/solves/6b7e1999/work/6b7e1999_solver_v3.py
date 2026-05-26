#!/usr/bin/env python3

import json
import copy

def solve(grid):
    """
    ARC-AGI puzzle 6b7e1999 solver - Point symmetry completion around centers
    
    Rule: Find cells with value 7 (when 7 is not background) and complete
    point symmetry (180-degree rotation) around each center.
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
    else:
        # When 7 is background, we need to find other centers
        # Look for distinct non-7 patterns that might indicate centers
        for r in range(1, H-1):
            for c in range(1, W-1):
                if grid[r][c] != 7:
                    # Check if this could be a center of a local pattern
                    non_bg_count = 0
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != 7:
                                non_bg_count += 1
                    if non_bg_count >= 3:  # Has enough pattern around it
                        centers.append((r, c))
        
        # Remove duplicates and keep only the most promising centers
        unique_centers = []
        for center in centers:
            too_close = False
            for existing in unique_centers:
                if abs(center[0] - existing[0]) <= 2 and abs(center[1] - existing[1]) <= 2:
                    too_close = True
                    break
            if not too_close:
                unique_centers.append(center)
        centers = unique_centers[:5]  # Limit to 5 centers max
    
    # For each center, complete point symmetry
    for center_r, center_c in centers:
        # Define region around center
        radius = 3  # Use 7x7 region
        
        # Multiple passes to ensure all symmetries are completed
        for pass_num in range(3):
            changed = False
            
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    r, c = center_r + dr, center_c + dc
                    
                    # Skip if out of bounds
                    if not (0 <= r < H and 0 <= c < W):
                        continue
                    
                    # Find point-symmetric position
                    sym_r = 2 * center_r - r
                    sym_c = 2 * center_c - c
                    
                    # Skip if symmetric position is out of bounds
                    if not (0 <= sym_r < H and 0 <= sym_c < W):
                        continue
                    
                    # If current position has non-background and symmetric position is background
                    if result[r][c] != bg_color and result[sym_r][sym_c] == bg_color:
                        result[sym_r][sym_c] = result[r][c]
                        changed = True
                    
                    # Also check the reverse
                    elif result[sym_r][sym_c] != bg_color and result[r][c] == bg_color:
                        result[r][c] = result[sym_r][sym_c]
                        changed = True
            
            if not changed:
                break
    
    return result

def test_solver():
    """Test the solver on training examples"""
    with open('/Users/evanpieser/apr12_tasks/6b7e1999.json', 'r') as f:
        task = json.load(f)
    
    print("Testing final point symmetry solver...")
    
    all_correct = True
    for i, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        predicted_output = solve(input_grid)
        
        correct = predicted_output == expected_output
        all_correct = all_correct and correct
        
        print(f"Train {i+1}: {'✓' if correct else '✗'}")
        
        if not correct:
            print("  Differences:")
            diff_count = 0
            for r in range(len(expected_output)):
                for c in range(len(expected_output[0])):
                    if predicted_output[r][c] != expected_output[r][c]:
                        print(f"    ({r},{c}): expected {expected_output[r][c]}, got {predicted_output[r][c]}")
                        diff_count += 1
                        if diff_count >= 5:  # Show only first 5 differences
                            if diff_count < sum(1 for rr in range(len(expected_output)) 
                                              for cc in range(len(expected_output[0]))
                                              if predicted_output[rr][cc] != expected_output[rr][cc]):
                                print("    ... (more differences)")
                            break
                if diff_count >= 5:
                    break
    
    print(f"\nOverall: {'All correct!' if all_correct else 'Some failures'}")
    return all_correct

if __name__ == '__main__':
    test_solver()