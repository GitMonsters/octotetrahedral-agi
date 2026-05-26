#!/usr/bin/env python3

import json
import numpy as np

def analyze_pair3_exact_pattern():
    with open('/Users/evanpieser/apr12_tasks/651b2ee5.json', 'r') as f:
        task = json.load(f)
    
    pair = task['train'][2]
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    print("=== EXACT ANALYSIS OF PAIR 3 PATTERN ===")
    print(f"Shape: {output_grid.shape}")
    
    h, w = output_grid.shape
    
    # Print the actual pattern with coordinates
    print("\nActual output with coordinates:")
    print("   ", end="")
    for x in range(w):
        print(f"{x:2d}", end=" ")
    print()
    
    for y in range(h):
        print(f"{y:2d}:", end="")
        for x in range(w):
            value = output_grid[y, x]
            if value == 6:  # marker color
                print(f" 6", end=" ")
            else:
                print(f" .", end=" ")
        print()
    
    # Find the exact pattern for marker positions
    marker_positions = []
    for y in range(h):
        for x in range(w):
            if output_grid[y, x] == 6:
                marker_positions.append((y, x))
    
    print(f"\nMarker positions: {len(marker_positions)} total")
    
    # Let's look for periodicity
    print(f"\nLooking for patterns in first few rows:")
    for y in range(min(8, h)):
        row_markers = [x for x in range(w) if output_grid[y, x] == 6]
        print(f"Row {y}: markers at columns {row_markers}")
    
    # Maybe it's a shifted/offset checkerboard
    print(f"\nTesting different offsets:")
    for offset_y in range(3):
        for offset_x in range(3):
            predicted = np.ones((h, w), dtype=int)
            for y in range(h):
                for x in range(w):
                    # Test shifted checkerboard
                    if ((y + offset_y) + (x + offset_x)) % 2 == 0:
                        predicted[y, x] = 6
            
            differences = np.sum(predicted != output_grid)
            if differences <= 15:  # Show promising ones
                accuracy = (h * w - differences) / (h * w)
                print(f"  Offset ({offset_y},{offset_x}): {differences} errors, {accuracy:.3f} accuracy")
    
    # Let's try a different approach - maybe it's based on distance from corners
    print(f"\nTesting corner-based patterns:")
    
    # Pattern might be based on distance from specific points
    # Let's see if it's related to the marker positions in input
    marker_input_positions = []
    for y in range(h):
        for x in range(w):
            if input_grid[y, x] == 6:  # Find marker positions in input
                marker_input_positions.append((y, x))
    
    print(f"Input marker positions: {marker_input_positions}")
    
    # Maybe the pattern is related to these positions
    if len(marker_input_positions) >= 2:
        y1, x1 = marker_input_positions[0]  # (0, 4)
        y2, x2 = marker_input_positions[1]  # (14, 4)
        
        print(f"\nTrying pattern based on distance to input markers:")
        predicted = np.ones((h, w), dtype=int)
        for y in range(h):
            for x in range(w):
                # Try various distance-based rules
                dist1 = abs(y - y1) + abs(x - x1)  # Manhattan distance to first marker
                dist2 = abs(y - y2) + abs(x - x2)  # Manhattan distance to second marker
                
                # Test various conditions
                if (dist1 % 2 == 0) or (dist2 % 2 == 0):
                    predicted[y, x] = 6
        
        differences = np.sum(predicted != output_grid)
        accuracy = (h * w - differences) / (h * w)
        print(f"  Distance-based: {differences} errors, {accuracy:.3f} accuracy")

if __name__ == "__main__":
    analyze_pair3_exact_pattern()