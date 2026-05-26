#!/usr/bin/env python3

import json
import numpy as np

# Load the task data
with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
    task_data = json.load(f)

def analyze_grid_structure(grid):
    """Analyze grid to find separator lines"""
    rows, cols = len(grid), len(grid[0])
    
    # Find horizontal separators (rows that are all the same value)
    h_seps = []
    for r in range(rows):
        row_values = set(grid[r])
        if len(row_values) == 1:
            h_seps.append(r)
    
    # Find vertical separators (columns that are all the same value)
    v_seps = []
    for c in range(cols):
        col_values = set(grid[r][c] for r in range(rows))
        if len(col_values) == 1:
            v_seps.append(c)
    
    return h_seps, v_seps

print("=== DETAILED GRID STRUCTURE AND PATTERN ANALYSIS ===")

for i, example in enumerate(task_data['train']):
    print(f"\n--- Training Pair {i+1} ---")
    input_grid = example['input']
    output_grid = example['output']
    
    # Analyze structure
    h_seps, v_seps = analyze_grid_structure(input_grid)
    print(f"Grid size: {len(input_grid)}x{len(input_grid[0])}")
    print(f"Horizontal separators at rows: {h_seps}")
    print(f"Vertical separators at cols: {v_seps}")
    
    # Determine tile dimensions
    if h_seps and v_seps:
        tile_height = h_seps[0] if h_seps[0] > 0 else h_seps[1] - h_seps[0] - 1
        tile_width = v_seps[0] if v_seps[0] > 0 else v_seps[1] - v_seps[0] - 1
        print(f"Tile dimensions: {tile_height}x{tile_width}")
        
        # Extract expected tile pattern (from top-left tile)
        expected_tile = []
        for r in range(tile_height):
            expected_tile.append(input_grid[r][:tile_width])
        
        print("Expected tile pattern:")
        for row in expected_tile:
            print(f"  {row}")
        
        # Find all anomalies in the input grid
        print("\nFinding anomalies in input...")
        anomalies = []
        
        # Check each tile position
        tile_rows = len(h_seps) + 1
        tile_cols = len(v_seps) + 1
        
        for tr in range(tile_rows):
            for tc in range(tile_cols):
                # Calculate grid position for this tile
                start_r = tr * (tile_height + 1)
                start_c = tc * (tile_width + 1)
                
                # Skip if we're past the grid bounds
                if start_r >= len(input_grid) or start_c >= len(input_grid[0]):
                    continue
                
                # Check each cell in this tile
                for dr in range(tile_height):
                    for dc in range(tile_width):
                        gr, gc = start_r + dr, start_c + dc
                        if gr < len(input_grid) and gc < len(input_grid[0]):
                            expected = expected_tile[dr][dc]
                            actual = input_grid[gr][gc]
                            if actual != expected:
                                anomalies.append({
                                    'grid_pos': (gr, gc),
                                    'tile_pos': (dr, dc),
                                    'tile_idx': (tr, tc),
                                    'expected': expected,
                                    'actual': actual
                                })
        
        print(f"Found {len(anomalies)} anomalies in input:")
        for anom in anomalies:
            print(f"  Grid({anom['grid_pos'][0]}, {anom['grid_pos'][1]}) in tile({anom['tile_idx'][0]}, {anom['tile_idx'][1]}) at pos({anom['tile_pos'][0]}, {anom['tile_pos'][1]}): {anom['actual']} (expected {anom['expected']})")
    
    # Compare input to output
    print(f"\nChanges from input to output:")
    changes = []
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r][c] != output_grid[r][c]:
                changes.append((r, c, input_grid[r][c], output_grid[r][c]))
    
    print(f"Found {len(changes)} changes:")
    for change in changes:
        print(f"  Grid({change[0]}, {change[1]}): {change[2]} → {change[3]}")