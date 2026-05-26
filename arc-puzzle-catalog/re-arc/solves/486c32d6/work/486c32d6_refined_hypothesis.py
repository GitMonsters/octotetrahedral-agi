#!/usr/bin/env python3

# REFINED HYPOTHESIS for ARC-AGI Task 486c32d6
# =============================================

"""
CORRECTED UNDERSTANDING:

The task is to COMPLETE the repeating tile pattern by filling in missing expected values.

RULE:
1. Identify the expected repeating tile pattern
2. For each tile position, if there are anomalous values (different from expected), 
   propagate the EXPECTED value to all other tiles in the same tile-row at the same within-tile position
3. The expected values are determined from the COMPLETE tile pattern that appears somewhere in the grid

Let me reanalyze to find the complete reference pattern...
"""

import json

with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
    task_data = json.load(f)

def find_complete_tile_pattern(input_grid, h_seps, v_seps, tile_height, tile_width):
    """Find the most complete tile pattern by examining all tiles"""
    
    # Collect all tile instances
    tile_instances = []
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
                tile_instances.append(((tr, tc), tile))
    
    # Find the reference pattern by majority vote for each position
    reference = [[None for _ in range(tile_width)] for _ in range(tile_height)]
    
    for dr in range(tile_height):
        for dc in range(tile_width):
            # Collect all values at this position across tiles
            values = []
            for (tr, tc), tile in tile_instances:
                values.append(tile[dr][dc])
            
            # Find most common value (or use first occurrence pattern)
            from collections import Counter
            counter = Counter(values)
            reference[dr][dc] = counter.most_common(1)[0][0]
    
    return reference

def analyze_example(example_idx):
    example = task_data['train'][example_idx]
    input_grid = example['input']
    output_grid = example['output']
    
    print(f"\n=== Training Pair {example_idx + 1} ===")
    
    # Find structure
    h_seps = [r for r in range(len(input_grid)) if len(set(input_grid[r])) == 1]
    v_seps = [c for c in range(len(input_grid[0])) if len(set(input_grid[r][c] for r in range(len(input_grid)))) == 1]
    
    tile_height = h_seps[0] if h_seps[0] > 0 else h_seps[1] - h_seps[0] - 1
    tile_width = v_seps[0] if v_seps[0] > 0 else v_seps[1] - v_seps[0] - 1
    
    print(f"Tile size: {tile_height}x{tile_width}")
    print(f"H separators: {h_seps}, V separators: {v_seps}")
    
    # Find complete reference pattern
    reference = find_complete_tile_pattern(input_grid, h_seps, v_seps, tile_height, tile_width)
    print(f"Reference tile pattern:")
    for row in reference:
        print(f"  {row}")
    
    # Find changes
    changes = []
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r][c] != output_grid[r][c]:
                changes.append((r, c, input_grid[r][c], output_grid[r][c]))
    
    print(f"\nChanges ({len(changes)}):")
    for change in changes:
        r, c, old_val, new_val = change
        
        # Find tile coordinates
        tile_r = sum(1 for sep in h_seps if sep < r)
        tile_c = sum(1 for sep in v_seps if sep < c)
        within_r = r - tile_r * (tile_height + 1)
        within_c = c - tile_c * (tile_width + 1)
        
        expected = reference[within_r][within_c]
        
        print(f"  ({r},{c}) tile({tile_r},{tile_c}) pos({within_r},{within_c}): {old_val}→{new_val} (ref:{expected})")
        
        if new_val == expected:
            print(f"    ✓ FILLS EXPECTED VALUE")
        else:
            print(f"    ✗ NOT MATCHING EXPECTED")

# Analyze all examples
for i in range(3):
    analyze_example(i)