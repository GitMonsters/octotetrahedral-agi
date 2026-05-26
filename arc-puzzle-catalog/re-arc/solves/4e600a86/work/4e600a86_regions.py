#!/usr/bin/env python3
"""
Alternative approach: Maybe it's about connected components or flood fill
"""
import json
import numpy as np

def load_task(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def find_connected_regions(grid, target_value):
    """Find connected regions of a specific value using flood fill"""
    visited = set()
    regions = []
    h, w = len(grid), len(grid[0])
    
    def flood_fill(start_r, start_c):
        stack = [(start_r, start_c)]
        region = []
        
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            if not (0 <= r < h and 0 <= c < w):
                continue
            if grid[r][c] != target_value:
                continue
                
            visited.add((r, c))
            region.append((r, c))
            
            # Add neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((r + dr, c + dc))
        
        return region
    
    for r in range(h):
        for c in range(w):
            if (r, c) not in visited and grid[r][c] == target_value:
                region = flood_fill(r, c)
                if region:
                    regions.append(region)
    
    return regions

def analyze_regions(pair_num, input_grid, output_grid):
    print(f"\n=== REGION ANALYSIS PAIR {pair_num} ===")
    
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)
    
    # Find background
    unique, counts = np.unique(input_arr, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    
    print(f"Background color: {bg_color}")
    
    # Find what changed
    changes = np.where(input_arr != output_arr)
    change_positions = list(zip(changes[0], changes[1]))
    print(f"Number of changes: {len(change_positions)}")
    
    if len(change_positions) == 0:
        print("No changes")
        return
    
    # Analyze pattern regions
    pattern_regions = find_connected_regions(input_grid, bg_color)
    print(f"Background regions: {len(pattern_regions)}")
    
    # Check if changes form connected regions
    change_grid = np.zeros_like(input_arr)
    for r, c in change_positions:
        change_grid[r, c] = 1
    
    change_regions = find_connected_regions(change_grid.tolist(), 1)
    print(f"Change regions: {len(change_regions)}")
    
    for i, region in enumerate(change_regions):
        print(f"  Change region {i+1}: {len(region)} cells")
        if len(region) <= 10:
            print(f"    Positions: {region}")

def main():
    task = load_task('/Users/evanpieser/apr12_tasks/4e600a86.json')
    
    for i, pair in enumerate(task['train']):
        analyze_regions(i + 1, pair['input'], pair['output'])

if __name__ == '__main__':
    main()