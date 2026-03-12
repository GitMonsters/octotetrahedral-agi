#!/usr/bin/env python3
"""
ARC-AGI task 37d3e8b2 solver.

Pattern: Regions of 8s are colored based on their position in horizontal bands.
- Regions in upper rows get one color (1, 2, or 3)
- Regions in lower rows get another color (2, 7, or 3)
The exact mapping seems to be: group regions by row-based bands and assign colors sequentially.
"""

import sys
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy input
    
    # Find all connected regions of 8s
    visited = [[False] * w for _ in range(h)]
    regions = []
    
    def dfs(r, c, region):
        if r < 0 or r >= h or c < 0 or c >= w:
            return
        if visited[r][c] or grid[r][c] != 8:
            return
        visited[r][c] = True
        region.append((r, c))
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc, region)
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 8 and not visited[r][c]:
                region = []
                dfs(r, c, region)
                if region:
                    regions.append(region)
    
    if not regions:
        return result
    
    # Assign colors based on hole count and position
    # Group regions by the number of "holes" (empty cells in bounding box)
    hole_to_regions = {}
    for region_idx, region in enumerate(regions):
        min_r = min(rr for rr, cc in region)
        max_r = max(rr for rr, cc in region)
        min_c = min(cc for rr, cc in region)
        max_c = max(cc for rr, cc in region)
        
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        total = height * width
        holes = total - len(region)
        
        if holes not in hole_to_regions:
            hole_to_regions[holes] = []
        hole_to_regions[holes].append((region_idx, min_r, region))
    
    # Assign colors - use a greedy approach
    # First, try grouping by row position (upper vs lower half)
    color_assignment = {}
    
    # Sort regions by their top row
    sorted_regions = sorted(enumerate(regions), key=lambda x: min(rr for rr, cc in x[1]))
    
    # Get distinct colors that should appear in output (excluding 0)
    available_colors = [1, 2, 3, 7]
    
    # Heuristic: try splitting at certain row thresholds
    mid_row = h // 2
    
    # Assign colors based on position bands
    color_idx = 0
    current_band = -1
    band_colors = {}
    
    for region_idx, region in sorted_regions:
        min_r = min(rr for rr, cc in region)
        
        # Determine which band this region is in
        if min_r < mid_row:
            band = 0
        else:
            band = 1
        
        # If we moved to a new band, potentially cycle color
        if band != current_band:
            current_band = band
            if band not in band_colors:
                band_colors[band] = available_colors[color_idx % len(available_colors)]
                color_idx += 1
        
        color = band_colors[band]
        color_assignment[region_idx] = color
    
    # Apply coloring
    for region_idx, region in enumerate(regions):
        color = color_assignment.get(region_idx, 1)
        for r, c in region:
            result[r][c] = color
    
    return result


if __name__ == "__main__":
    # Load task JSON
    task_path = sys.argv[1] if len(sys.argv) > 1 else "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/37d3e8b2.json"
    task_path = task_path.replace("~", "/Users/evanpieser")
    
    with open(task_path) as f:
        task = json.load(f)
    
    # Test on all training examples
    all_pass = True
    for idx, example in enumerate(task['train']):
        inp = example['input']
        expected = example['output']
        result = solve(inp)
        
        match = result == expected
        status = "PASS" if match else "FAIL"
        print(f"Train {idx}: {status}")
        
        if not match:
            all_pass = False
            # Show first diff
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  Diff at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
                        break
    
    if all_pass:
        print("\nAll training examples passed!")
    else:
        print("\nSome training examples failed.")
        sys.exit(1)
