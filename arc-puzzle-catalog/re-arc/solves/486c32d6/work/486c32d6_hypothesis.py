#!/usr/bin/env python3

# HYPOTHESIS for ARC-AGI Task 486c32d6
# ==================================

"""
GRID STRUCTURE ANALYSIS:
- Each grid has a repeating tile pattern separated by uniform rows/columns
- Separator rows/columns contain a single value repeated across the entire row/column
- The grids are divided into rectangular tiles of consistent size within each example

OBSERVED PATTERN:
1. Each grid contains anomalous pixels that deviate from the expected tile pattern
2. These anomalous pixels propagate horizontally across their row to fill ALL positions 
   in that row that should contain the same value as the anomaly
3. The propagation only occurs within the same tile row, and only for the specific 
   tile position (within-tile coordinates) that contains the anomaly

DETAILED HYPOTHESIS:
- When an anomalous pixel is found at position (tile_row, tile_col, within_row, within_col)
- The anomalous value propagates to ALL tiles in the same tile_row
- Specifically, it fills position (within_row, within_col) in all tiles of that tile_row
- The propagation is HORIZONTAL ONLY - across columns, not across rows

EXAMPLES FROM TRAINING DATA:

Training 1:
- Tile pattern 3x3, anomalies at (1,2) position have value 5 in first tile only
- Expected: All tiles should have 5 at position (1,2), but most have 2
- Output: Row 1 gets 5s propagated to positions (1,6), (1,10), (1,14) - all (1,2) positions across tile row 0

Training 2: 
- Tile pattern 3x3, anomalies at (1,0) and (2,0) positions
- Anomaly 9 at (1,0) in some tiles, anomaly 2 at (2,0) in some tiles  
- Output: Missing 9s and 2s get propagated horizontally to fill their respective positions

Training 3:
- Tile pattern 3x2, various anomalies
- Same horizontal propagation pattern within tile rows
"""

print("HYPOTHESIS VERIFICATION:")
print("=======================")

# Let me verify this hypothesis by checking if it explains all the observed changes

import json

with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
    task_data = json.load(f)

def verify_hypothesis(example_idx):
    example = task_data['train'][example_idx]
    input_grid = example['input']
    output_grid = example['output']
    
    print(f"\nVerifying Training Pair {example_idx + 1}:")
    
    # Find structure
    h_seps = [r for r in range(len(input_grid)) if len(set(input_grid[r])) == 1]
    v_seps = [c for c in range(len(input_grid[0])) if len(set(input_grid[r][c] for r in range(len(input_grid)))) == 1]
    
    tile_height = h_seps[0] if h_seps[0] > 0 else h_seps[1] - h_seps[0] - 1
    tile_width = v_seps[0] if v_seps[0] > 0 else v_seps[1] - v_seps[0] - 1
    
    # Extract expected tile
    expected_tile = []
    for r in range(tile_height):
        expected_tile.append(input_grid[r][:tile_width])
    
    print(f"Tile size: {tile_height}x{tile_width}")
    print(f"Expected tile pattern: {expected_tile}")
    
    # Find anomalies and check if propagation explains outputs
    changes = []
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r][c] != output_grid[r][c]:
                changes.append((r, c, input_grid[r][c], output_grid[r][c]))
    
    print(f"Changes in output: {len(changes)}")
    for change in changes:
        print(f"  ({change[0]}, {change[1]}): {change[2]} → {change[3]}")
    
    # Verify propagation logic
    print("Checking if changes follow propagation rule...")
    
    all_match = True
    for change in changes:
        r, c, old_val, new_val = change
        
        # Skip separator positions
        if r in h_seps or c in v_seps:
            continue
            
        # Find tile position
        tile_r = sum(1 for sep in h_seps if sep < r)
        tile_c = sum(1 for sep in v_seps if sep < c)
        
        # Find position within tile
        within_r = r - tile_r * (tile_height + 1)
        within_c = c - tile_c * (tile_width + 1)
        
        print(f"  Change at ({r},{c}) is in tile({tile_r},{tile_c}) at within-tile pos({within_r},{within_c})")
        print(f"    Expected at this pos: {expected_tile[within_r][within_c]}, got {new_val}")
        
        # Check if there's an anomaly elsewhere in the same tile row at the same within-tile position
        found_source = False
        for other_tile_c in range(len(v_seps) + 1):
            if other_tile_c == tile_c:
                continue
                
            other_c = other_tile_c * (tile_width + 1) + within_c
            other_r = r  # Same row
            
            if other_r < len(input_grid) and other_c < len(input_grid[0]):
                if input_grid[other_r][other_c] == new_val and input_grid[other_r][other_c] != expected_tile[within_r][within_c]:
                    print(f"    Found source anomaly at ({other_r},{other_c}) with value {new_val}")
                    found_source = True
                    break
        
        if not found_source:
            print(f"    ERROR: No source anomaly found for this propagation!")
            all_match = False
    
    return all_match

# Verify hypothesis on all training examples
for i in range(3):
    result = verify_hypothesis(i)
    print(f"Training {i+1} matches hypothesis: {result}")