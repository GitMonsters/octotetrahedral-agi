#!/usr/bin/env python3

import json
import copy

def solve(grid):
    """
    ARC-AGI puzzle 6b7e1999 solver - Symmetry completion around centers
    
    The rule: Find rectangular regions with patterns and complete them 
    by reflecting cells to create symmetric patterns around center points.
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
    
    def find_pattern_regions():
        """Find rectangular regions with patterns to complete"""
        regions = []
        
        # Look for various sizes of rectangular regions
        for size in range(4, 8):  # 4x4 to 7x7
            for top_r in range(H - size + 1):
                for left_c in range(W - size + 1):
                    # Count non-background cells in this region
                    non_bg_count = 0
                    for r in range(top_r, top_r + size):
                        for c in range(left_c, left_c + size):
                            if grid[r][c] != bg_color:
                                non_bg_count += 1
                    
                    # If region has a reasonable number of pattern cells
                    if 3 <= non_bg_count <= size * size * 0.6:
                        regions.append((top_r, left_c, size, size, non_bg_count))
        
        # Sort by density of non-background cells
        regions.sort(key=lambda x: x[4], reverse=True)
        return regions
    
    def complete_symmetry(top_r, left_c, height, width):
        """Complete symmetry in a rectangular region"""
        center_r = top_r + height // 2
        center_c = left_c + width // 2
        
        changed = True
        iterations = 0
        while changed and iterations < 5:
            changed = False
            iterations += 1
            
            # Pass 1: Horizontal reflections
            for r in range(top_r, top_r + height):
                for c in range(left_c, left_c + width):
                    if result[r][c] != bg_color:
                        # Find horizontal mirror position
                        mirror_c = 2 * center_c - c
                        if left_c <= mirror_c < left_c + width:
                            if result[r][mirror_c] == bg_color:
                                result[r][mirror_c] = result[r][c]
                                changed = True
            
            # Pass 2: Vertical reflections  
            for r in range(top_r, top_r + height):
                for c in range(left_c, left_c + width):
                    if result[r][c] != bg_color:
                        # Find vertical mirror position
                        mirror_r = 2 * center_r - r
                        if top_r <= mirror_r < top_r + height:
                            if result[mirror_r][c] == bg_color:
                                result[mirror_r][c] = result[r][c]
                                changed = True
            
            # Pass 3: Pattern completion within rows
            for r in range(top_r, top_r + height):
                # Look for A_A pattern and extend to A_A_A
                for c in range(left_c, left_c + width - 4):
                    if (result[r][c] != bg_color and result[r][c+1] == bg_color and 
                        result[r][c+2] != bg_color and result[r][c] == result[r][c+2] and
                        result[r][c+3] == bg_color and result[r][c+4] == bg_color):
                        result[r][c+4] = result[r][c]
                        changed = True
    
    # Find regions and complete them
    regions = find_pattern_regions()
    
    # Track used positions to avoid overlaps
    used = set()
    
    for region in regions:
        top_r, left_c, height, width, _ = region
        
        # Check if region overlaps with used areas
        overlap = False
        for r in range(top_r, top_r + height):
            for c in range(left_c, left_c + width):
                if (r, c) in used:
                    overlap = True
                    break
            if overlap:
                break
        
        if not overlap:
            # Mark as used
            for r in range(top_r, top_r + height):
                for c in range(left_c, left_c + width):
                    used.add((r, c))
            
            # Complete symmetry in this region
            complete_symmetry(top_r, left_c, height, width)
    
    return result

def test_solver():
    """Test the solver on training examples"""
    with open('/Users/evanpieser/apr12_tasks/6b7e1999.json', 'r') as f:
        task = json.load(f)
    
    print("Testing solver on training examples...")
    
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
            for r in range(len(expected_output)):
                for c in range(len(expected_output[0])):
                    if predicted_output[r][c] != expected_output[r][c]:
                        print(f"    ({r},{c}): expected {expected_output[r][c]}, got {predicted_output[r][c]}")
    
    print(f"\nOverall: {'All correct!' if all_correct else 'Some failures'}")
    return all_correct

if __name__ == '__main__':
    test_solver()