#!/usr/bin/env python3

import json
import numpy as np

def debug_train_pair_2():
    with open('/Users/evanpieser/apr12_tasks/651b2ee5.json', 'r') as f:
        task = json.load(f)
    
    # Check train pair 2 
    pair = task['train'][1]
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    print("=== DEBUGGING TRAIN PAIR 2 ===")
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
        print(f"Row {y}: {row_str}")
    
    print(f"\nMY 8-DIAGONAL PREDICTION:")
    my_output = np.ones((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            if (y + x) % 8 == 0 or (y - x) % 8 == 0:
                my_output[y, x] = marker_color
    
    for y in range(h):
        row_str = ""
        for x in range(w):
            row_str += f"{my_output[y,x]:2d} "
        print(f"Row {y}: {row_str}")
    
    print(f"\nCOMPARING ACTUAL vs PREDICTED:")
    differences = []
    for y in range(h):
        for x in range(w):
            if output_grid[y, x] != my_output[y, x]:
                differences.append((y, x, output_grid[y, x], my_output[y, x]))
    
    print(f"Found {len(differences)} differences:")
    for y, x, actual, predicted in differences[:20]:
        print(f"  ({y},{x}): actual={actual}, predicted={predicted}")
    
    # Let's try different patterns
    print(f"\nTrying 4-unit diagonal pattern:")
    pattern4_output = np.ones((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            if (y + x) % 4 == 0 or (y - x) % 4 == 0:
                pattern4_output[y, x] = marker_color
    
    differences4 = []
    for y in range(h):
        for x in range(w):
            if output_grid[y, x] != pattern4_output[y, x]:
                differences4.append((y, x, output_grid[y, x], pattern4_output[y, x]))
    
    print(f"4-unit pattern has {len(differences4)} differences")
    
    # Let's see what the actual positions are
    print(f"\nActual positions where marker color {marker_color} appears:")
    marker_positions = []
    for y in range(h):
        for x in range(w):
            if output_grid[y, x] == marker_color:
                marker_positions.append((y, x))
    
    print(f"Marker positions: {marker_positions}")
    
    # Check pattern
    print(f"\nAnalyzing pattern in marker positions:")
    for y, x in marker_positions:
        print(f"  ({y},{x}): y+x={y+x}, y-x={y-x}, (y+x)%4={(y+x)%4}, (y-x)%4={(y-x)%4}")

if __name__ == "__main__":
    debug_train_pair_2()