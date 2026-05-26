#!/usr/bin/env python3

import json
import numpy as np

def analyze_patterns():
    with open('/Users/evanpieser/apr12_tasks/651b2ee5.json', 'r') as f:
        task = json.load(f)
    
    print("=== DETAILED PATTERN ANALYSIS ===\n")
    
    for i, pair in enumerate(task['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"TRAIN PAIR {i+1}:")
        print(f"Input shape: {input_grid.shape}")
        
        # Find marker positions and colors
        unique_colors = np.unique(input_grid)
        print(f"Input colors: {unique_colors}")
        
        # Determine background color (most frequent)
        color_counts = {}
        for color in unique_colors:
            color_counts[color] = np.sum(input_grid == color)
        
        background_color = max(color_counts, key=color_counts.get)
        marker_colors = [c for c in unique_colors if c != background_color]
        
        print(f"Background color: {background_color} (count: {color_counts[background_color]})")
        print(f"Marker colors: {marker_colors}")
        
        # Find exact marker positions
        marker_positions = []
        for color in marker_colors:
            positions = np.where(input_grid == color)
            for y, x in zip(positions[0], positions[1]):
                marker_positions.append((y, x, color))
        
        print(f"Marker positions: {marker_positions}")
        
        # Analyze output pattern - focus on first few rows/columns to find the diagonal pattern
        print(f"\nOutput analysis (first 8x8 if available):")
        h, w = output_grid.shape
        max_show = min(8, h, w)
        
        for y in range(max_show):
            row_str = ""
            for x in range(max_show):
                row_str += f"{output_grid[y,x]:2d} "
            print(f"Row {y}: {row_str}")
        
        # Look for diagonal patterns
        print(f"\nDiagonal pattern analysis:")
        
        # Check for repeating patterns
        if h >= 4 and w >= 4:
            print("Checking for 2x2 repeating pattern...")
            for start_y in range(2):
                for start_x in range(2):
                    pattern_2x2 = output_grid[start_y:start_y+2, start_x:start_x+2]
                    print(f"  Pattern starting at ({start_y},{start_x}):")
                    for row in pattern_2x2:
                        print(f"    {' '.join(str(x) for x in row)}")
        
        print("-" * 50)

if __name__ == "__main__":
    analyze_patterns()