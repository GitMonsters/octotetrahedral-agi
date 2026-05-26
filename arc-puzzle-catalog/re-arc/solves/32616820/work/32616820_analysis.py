#!/usr/bin/env python3

import json

# Load task data
with open('/Users/evanpieser/apr12_tasks/32616820.json', 'r') as f:
    task = json.load(f)

print("=== ANALYZING CHANGES IN EACH TRAINING PAIR ===")

for i, pair in enumerate(task['train']):
    print(f"\n=== Training Pair {i+1} ===")
    input_grid = pair['input']
    output_grid = pair['output']
    
    height, width = len(input_grid), len(input_grid[0])
    
    # Find all changed cells
    changes = []
    for r in range(height):
        for c in range(width):
            if input_grid[r][c] != output_grid[r][c]:
                changes.append((r, c, input_grid[r][c], output_grid[r][c]))
    
    print(f"Total changes: {len(changes)}")
    
    # Group changes by input color (what was removed)
    from collections import defaultdict
    by_input_color = defaultdict(list)
    for r, c, inp, out in changes:
        by_input_color[inp].append((r, c, out))
    
    print("Changes grouped by input color:")
    for color, positions in by_input_color.items():
        print(f"  Color {color}: {len(positions)} cells changed")
        # Show a few examples
        for j, (r, c, new_val) in enumerate(positions[:5]):
            print(f"    ({r},{c}): {color} -> {new_val}")
        if len(positions) > 5:
            print(f"    ... and {len(positions)-5} more")
    
    # Look for corruption pattern - find large connected regions of same color
    print("\nLooking for corruption (large solid regions):")
    
    def find_connected_region(grid, start_r, start_c, target_color, visited):
        if (start_r < 0 or start_r >= len(grid) or 
            start_c < 0 or start_c >= len(grid[0]) or
            (start_r, start_c) in visited or
            grid[start_r][start_c] != target_color):
            return []
        
        visited.add((start_r, start_c))
        region = [(start_r, start_c)]
        
        # Check 4 directions
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            region.extend(find_connected_region(grid, start_r + dr, start_c + dc, target_color, visited))
        
        return region
    
    visited = set()
    large_regions = []
    
    for r in range(height):
        for c in range(width):
            if (r, c) not in visited:
                color = input_grid[r][c]
                region = find_connected_region(input_grid, r, c, color, visited)
                if len(region) >= 10:  # Large region threshold
                    large_regions.append((color, len(region), region[:5]))  # Show first 5 cells
    
    print("Large connected regions in input:")
    for color, size, sample_cells in large_regions:
        print(f"  Color {color}: {size} connected cells, sample: {sample_cells}")
        
    # Check if this is a repeating pattern
    print(f"\nLooking for repeating pattern:")
    for period in [6, 7]:
        print(f"Testing period {period}:")
        pattern_matches = True
        for r in range(height - period):
            if output_grid[r] != output_grid[r + period]:
                pattern_matches = False
                break
        if pattern_matches:
            print(f"  ✓ Perfect vertical repeat with period {period}")
        else:
            # Count matches
            total_matches = 0
            total_cells = 0
            for r in range(height - period):
                for c in range(width):
                    total_cells += 1
                    if output_grid[r][c] == output_grid[r + period][c]:
                        total_matches += 1
            match_rate = total_matches / total_cells if total_cells > 0 else 0
            print(f"  Partial match with period {period}: {match_rate:.2%}")

print("\n=== HYPOTHESIS VERIFICATION ===")
print("The pattern appears to be:")
print("1. Input has a repeating tile pattern corrupted by large solid color blocks")
print("2. Large blocks of same color (3=green, 7=orange) are corruption")
print("3. Output reconstructs the clean repeating pattern by removing corruption")