#!/usr/bin/env python3

import json
import sys

def analyze_changes(input_grid, output_grid):
    """Analyze what changed between input and output"""
    rows, cols = len(input_grid), len(input_grid[0])
    changes = []
    
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != output_grid[r][c]:
                changes.append((r, c, input_grid[r][c], output_grid[r][c]))
    
    return changes

def main():
    with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/7666fa5d.json', 'r') as f:
        data = json.load(f)
    
    print("=== ANALYZING TRAINING EXAMPLES ===")
    
    for i, example in enumerate(data['train']):
        print(f"\n--- Example {i+1} ---")
        input_grid = example['input']
        output_grid = example['output']
        
        changes = analyze_changes(input_grid, output_grid)
        print(f"Total changes: {len(changes)}")
        
        # Find marker value
        marker_val = None
        for r in range(len(input_grid)):
            for c in range(len(input_grid[0])):
                if input_grid[r][c] != 8:
                    marker_val = input_grid[r][c]
                    break
            if marker_val:
                break
        
        print(f"Marker value: {marker_val}")
        
        # Show some changes
        if changes:
            print("Sample changes (r, c, old, new):")
            for change in changes[:10]:
                print(f"  {change}")
                
        # Find marker positions
        markers = []
        for r in range(len(input_grid)):
            for c in range(len(input_grid[0])):
                if input_grid[r][c] != 8:
                    markers.append((r, c))
        
        print(f"Marker positions: {markers}")
        
        # Analyze the pattern of 2's in the output
        twos = []
        for r in range(len(output_grid)):
            for c in range(len(output_grid[0])):
                if output_grid[r][c] == 2:
                    twos.append((r, c))
        
        print(f"Positions filled with 2: {len(twos)} cells")
        if twos:
            print(f"Sample 2 positions: {twos[:10]}")

if __name__ == "__main__":
    main()