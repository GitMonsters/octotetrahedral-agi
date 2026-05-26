#!/usr/bin/env python3

import json
import numpy as np

def test_unified_pattern():
    with open('/Users/evanpieser/apr12_tasks/651b2ee5.json', 'r') as f:
        task = json.load(f)
    
    print("=== TESTING UNIFIED PATTERN HYPOTHESIS ===\n")
    
    # My hypothesis: The pattern depends on the grid dimensions
    # - Large grids (like 9x17): use 8-diagonal pattern 
    # - Medium grids (like 5x15): use 4-diagonal pattern
    # - Small/square-ish grids (like 15x5): use checkerboard pattern
    
    for i, pair in enumerate(task['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"TRAIN PAIR {i+1}: {input_grid.shape}")
        
        # Find marker colors
        unique_colors = np.unique(input_grid)
        color_counts = {}
        for color in unique_colors:
            color_counts[color] = np.sum(input_grid == color)
        
        background_color = max(color_counts, key=color_counts.get)
        marker_colors = [c for c in unique_colors if c != background_color]
        marker_color = marker_colors[0]
        
        h, w = output_grid.shape
        
        # Try different patterns based on dimensions
        max_dim = max(h, w)
        
        if max_dim >= 15:
            # Large grids: 8-diagonal pattern
            print(f"  Using 8-diagonal pattern (max_dim = {max_dim})")
            predicted = np.ones((h, w), dtype=int)
            for y in range(h):
                for x in range(w):
                    if (y + x) % 8 == 0 or (y - x) % 8 == 0:
                        predicted[y, x] = marker_color
        
        elif max_dim >= 10:
            # Medium grids: 4-diagonal pattern
            print(f"  Using 4-diagonal pattern (max_dim = {max_dim})")
            predicted = np.ones((h, w), dtype=int)
            for y in range(h):
                for x in range(w):
                    if (y + x) % 4 == 0 or (y - x) % 4 == 0:
                        predicted[y, x] = marker_color
        
        else:
            # Small grids: checkerboard pattern
            print(f"  Using checkerboard pattern (max_dim = {max_dim})")
            predicted = np.ones((h, w), dtype=int)
            for y in range(h):
                for x in range(w):
                    if (y + x) % 2 == 0:
                        predicted[y, x] = marker_color
        
        differences = np.sum(predicted != output_grid)
        accuracy = (h * w - differences) / (h * w)
        
        print(f"  Accuracy: {accuracy:.3f} ({differences} differences)")
        
        if differences == 0:
            print(f"  ✓ PERFECT MATCH!")
        else:
            print(f"  ✗ Still has errors")
        
        print()

if __name__ == "__main__":
    test_unified_pattern()