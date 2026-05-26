#!/usr/bin/env python3

import json
import numpy as np

def debug_train_pair_3():
    with open('/Users/evanpieser/apr12_tasks/651b2ee5.json', 'r') as f:
        task = json.load(f)
    
    # Check train pair 3 
    pair = task['train'][2]
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    print("=== DEBUGGING TRAIN PAIR 3 ===")
    print(f"Input shape: {input_grid.shape}")
    print(f"Output shape: {output_grid.shape}")
    
    # Find marker colors
    unique_colors = np.unique(input_grid)
    print(f"Input colors: {unique_colors}")
    
    color_counts = {}
    for color in unique_colors:
        color_counts[color] = np.sum(input_grid == color)
    
    background_color = max(color_counts, key=color_counts.get)
    marker_colors = [c for c in unique_colors if c != background_color]
    marker_color = marker_colors[0]
    
    print(f"Background color: {background_color} (count: {color_counts[background_color]})")
    print(f"Marker color: {marker_color} (count: {color_counts[marker_color]})")
    
    h, w = output_grid.shape
    
    print(f"\nACTUAL OUTPUT GRID:")
    for y in range(h):
        row_str = ""
        for x in range(w):
            row_str += f"{output_grid[y,x]:2d} "
        print(f"Row {y:2d}: {row_str}")
    
    print(f"\nActual positions where marker color {marker_color} appears:")
    marker_positions = []
    for y in range(h):
        for x in range(w):
            if output_grid[y, x] == marker_color:
                marker_positions.append((y, x))
    
    print(f"Marker positions: {marker_positions}")
    
    # Check pattern - look at first few to understand
    print(f"\nAnalyzing pattern in marker positions:")
    for i, (y, x) in enumerate(marker_positions[:15]):
        print(f"  ({y},{x}): y+x={y+x}, y-x={y-x}, (y+x)%2={(y+x)%2}, (y-x)%2={(y-x)%2}, (y+x)%3={(y+x)%3}, (y-x)%3={(y-x)%3}")
    
    # Let's check if it follows a different rule - maybe specific period or offset
    print(f"\nTrying different patterns:")
    
    # Pattern based on position in grid
    for period in [2, 3, 4, 5]:
        predicted = np.ones((h, w), dtype=int)
        for y in range(h):
            for x in range(w):
                # Try different combinations
                if (y + x + 2) % period == 0 or (y - x + 2) % period == 0:
                    predicted[y, x] = marker_color
        
        differences = np.sum(predicted != output_grid)
        accuracy = (h * w - differences) / (h * w)
        if accuracy > 0.85:
            print(f"  Pattern (y+x+2) % {period} OR (y-x+2) % {period}: {accuracy:.3f}")
    
    # Check if it's a checkerboard with offset
    print(f"\nCheckerboard-like patterns:")
    for offset_y in range(3):
        for offset_x in range(3):
            predicted = np.ones((h, w), dtype=int)
            for y in range(h):
                for x in range(w):
                    if ((y + offset_y) + (x + offset_x)) % 2 == 0:
                        predicted[y, x] = marker_color
            
            differences = np.sum(predicted != output_grid)
            accuracy = (h * w - differences) / (h * w)
            if accuracy > 0.85:
                print(f"  Checkerboard offset ({offset_y},{offset_x}): {accuracy:.3f}")

if __name__ == "__main__":
    debug_train_pair_3()