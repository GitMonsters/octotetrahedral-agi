#!/usr/bin/env python3

import json
import numpy as np

def analyze_train_pair_1_detailed():
    with open('/Users/evanpieser/apr12_tasks/651b2ee5.json', 'r') as f:
        task = json.load(f)
    
    pair = task['train'][0]  # First pair
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    print("=== DETAILED ANALYSIS OF TRAIN PAIR 1 ===")
    print(f"Input shape: {input_grid.shape}")
    print(f"Output shape: {output_grid.shape}")
    
    # Find marker positions (color 0)
    marker_positions = []
    positions = np.where(input_grid == 0)
    for y, x in zip(positions[0], positions[1]):
        marker_positions.append((y, x))
    
    print(f"Marker positions (color 0): {marker_positions}")
    
    # Print every cell of the output to find the pattern
    print(f"\nFULL OUTPUT GRID (every cell):")
    h, w = output_grid.shape
    
    print("   ", end="")
    for x in range(w):
        print(f"{x:2d}", end=" ")
    print()
    
    for y in range(h):
        print(f"{y:2d}: ", end="")
        for x in range(w):
            value = output_grid[y, x]
            print(f"{value:2d}", end=" ")
        print()
    
    print(f"\nLooking for diagonal patterns...")
    
    # Check if there's a regular diagonal pattern
    print(f"\nChecking main diagonals:")
    
    # Look at positions where marker color (0) appears in output
    marker_output_positions = []
    positions = np.where(output_grid == 0)
    for y, x in zip(positions[0], positions[1]):
        marker_output_positions.append((y, x))
    
    print(f"Positions with marker color (0) in output: {marker_output_positions}")
    
    # Check if there's a pattern based on sum of coordinates
    print(f"\nAnalyzing coordinate patterns:")
    for y, x in marker_output_positions:
        print(f"Position ({y},{x}): y+x = {y+x}, y-x = {y-x}, y%8 = {y%8}, x%8 = {x%8}")

if __name__ == "__main__":
    analyze_train_pair_1_detailed()