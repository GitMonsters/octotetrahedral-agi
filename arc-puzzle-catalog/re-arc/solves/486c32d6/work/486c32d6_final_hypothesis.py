#!/usr/bin/env python3

# FINAL HYPOTHESIS for ARC-AGI Task 486c32d6
# ===========================================

"""
KEY INSIGHT: The pattern propagates anomalous values horizontally within the same tile row.

When an anomalous value appears at a specific within-tile position in one tile,
that same anomalous value should appear at the same within-tile position in ALL tiles 
of the same tile row.

The rule is: HORIZONTAL PROPAGATION of anomalous values within tile rows.
"""

import json

with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
    task_data = json.load(f)

def solve_example(example_idx):
    example = task_data['train'][example_idx]
    input_grid = [row[:] for row in example['input']]  # Copy input
    expected_output = example['output']
    
    print(f"\n=== Training Pair {example_idx + 1} ===")
    
    # Find structure
    h_seps = [r for r in range(len(input_grid)) if len(set(input_grid[r])) == 1]
    v_seps = [c for c in range(len(input_grid[0])) if len(set(input_grid[r][c] for r in range(len(input_grid)))) == 1]
    
    tile_height = h_seps[0] if h_seps[0] > 0 else h_seps[1] - h_seps[0] - 1  
    tile_width = v_seps[0] if v_seps[0] > 0 else v_seps[1] - v_seps[0] - 1
    
    print(f"Tile size: {tile_height}x{tile_width}")
    
    # Get the baseline tile (most common pattern)
    tile_instances = {}
    tile_rows = len(h_seps) + 1
    tile_cols = len(v_seps) + 1
    
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            start_r = tr * (tile_height + 1)  
            start_c = tc * (tile_width + 1)
            
            if start_r + tile_height <= len(input_grid) and start_c + tile_width <= len(input_grid[0]):
                tile = []
                for dr in range(tile_height):
                    row = []
                    for dc in range(tile_width):
                        gr, gc = start_r + dr, start_c + dc
                        row.append(input_grid[gr][gc])
                    tile.append(row)
                tile_instances[(tr, tc)] = tile
    
    # Identify anomalous values for each tile row and within-tile position
    output_grid = [row[:] for row in input_grid]  # Start with input
    
    for tr in range(tile_rows):
        print(f"\nProcessing tile row {tr}:")
        
        # Get all tiles in this row
        row_tiles = [(tc, tile_instances[(tr, tc)]) for tc in range(tile_cols) if (tr, tc) in tile_instances]
        
        if len(row_tiles) < 2:
            continue
            
        # For each within-tile position, find anomalous values and propagate them
        for within_r in range(tile_height):
            for within_c in range(tile_width):
                # Collect all values at this position across this tile row
                values_at_pos = []
                for tc, tile in row_tiles:
                    values_at_pos.append((tc, tile[within_r][within_c]))
                
                print(f"  Position ({within_r},{within_c}): {values_at_pos}")
                
                # Find unique values (potential anomalies)
                unique_values = list(set(val for tc, val in values_at_pos))
                
                if len(unique_values) > 1:
                    # There are different values - propagate the minority values
                    from collections import Counter
                    value_counts = Counter(val for tc, val in values_at_pos)
                    
                    # Find values that appear in minority (these are the anomalies to propagate)
                    for val, count in value_counts.items():
                        if count < len(row_tiles):  # Not in all tiles
                            print(f"    Propagating value {val} to all tiles in row {tr}")
                            
                            # Apply this value to all tiles in the row at this position
                            for tc, tile in row_tiles:
                                start_r = tr * (tile_height + 1)
                                start_c = tc * (tile_width + 1)
                                gr = start_r + within_r
                                gc = start_c + within_c
                                
                                if gr < len(output_grid) and gc < len(output_grid[0]):
                                    print(f"      Setting ({gr},{gc}) = {val}")
                                    output_grid[gr][gc] = val
    
    # Check if our output matches expected
    matches = True
    differences = []
    for r in range(len(expected_output)):
        for c in range(len(expected_output[0])):
            if output_grid[r][c] != expected_output[r][c]:
                differences.append((r, c, output_grid[r][c], expected_output[r][c]))
                matches = False
    
    print(f"\nResult: {'✓ MATCH' if matches else '✗ MISMATCH'}")
    if differences:
        print("Differences:")
        for diff in differences[:10]:
            print(f"  ({diff[0]},{diff[1]}): got {diff[2]}, expected {diff[3]}")
    
    return matches

# Test on all training examples
for i in range(3):
    solve_example(i)