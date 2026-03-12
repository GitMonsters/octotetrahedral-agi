#!/usr/bin/env python3
"""
Solver for ARC task 50aad11f.

Pattern: Extract all connected components of 6s and color them based on nearby color markers.
Output is all components stacked vertically, converted to 4x4 bounding boxes and recolored.
"""

import json
import sys
from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Extract connected components of 6s, identify associated colors,
    and output the components recolored and stacked vertically.
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    # Find all sixes and color markers
    sixes = []
    colors = {}  # color value -> (row, col) position
    
    for i in range(height):
        for j in range(width):
            if grid[i][j] == 6:
                sixes.append((i, j))
            elif grid[i][j] != 0:
                colors[grid[i][j]] = (i, j)
    
    # Extract connected components of 6s
    visited = set()
    components = []
    
    def get_component(start_r, start_c):
        """Extract a connected component using BFS."""
        comp = []
        q = deque([(start_r, start_c)])
        visited.add((start_r, start_c))
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < height and 0 <= nc < width and
                    (nr, nc) not in visited and grid[nr][nc] == 6):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return comp
    
    for r, c in sixes:
        if (r, c) not in visited:
            comp = get_component(r, c)
            components.append(comp)
    
    # For each component, extract its 4x4 bounding box as binary
    # Also track the component's top-left position for ordering
    component_grids = []
    for comp in components:
        min_r = min(cell[0] for cell in comp)
        max_r = max(cell[0] for cell in comp)
        min_c = min(cell[1] for cell in comp)
        max_c = max(cell[1] for cell in comp)
        
        # Extract bounding box
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        
        # Create binary grid for this component
        comp_set = set(comp)
        bin_grid = []
        for r in range(min_r, max_r + 1):
            row = []
            for c in range(min_c, max_c + 1):
                row.append(1 if (r, c) in comp_set else 0)
            bin_grid.append(row)
        component_grids.append((bin_grid, comp, min_r, min_c))
    
    # Determine stacking orientation: check if components are roughly aligned vertically
    # (similar starting rows) or horizontally aligned (different starting rows)
    min_rows = [x[2] for x in component_grids]
    row_spread = max(min_rows) - min(min_rows) if len(min_rows) > 1 else 0
    
    # If components are close in row position, use horizontal concatenation
    # Otherwise, use vertical stacking
    use_horizontal = row_spread <= 3 if len(component_grids) > 1 else False
    
    # Sort components appropriately
    if use_horizontal:
        # Sort by column position (left-to-right)
        component_grids.sort(key=lambda x: x[3])
    else:
        # Sort by row position (top-to-bottom)
        component_grids.sort(key=lambda x: (x[2], x[3]))
    
    # Match each component to a color based on closest color marker
    component_colors = [0] * len(component_grids)
    
    for color_val, (color_r, color_c) in colors.items():
        # Find closest component to this color marker
        min_dist = float('inf')
        closest_idx = -1
        
        for idx, (_, comp, _, _) in enumerate(component_grids):
            # Distance from color marker to component's center
            comp_center_r = sum(cell[0] for cell in comp) / len(comp)
            comp_center_c = sum(cell[1] for cell in comp) / len(comp)
            dist = (color_r - comp_center_r) ** 2 + (color_c - comp_center_c) ** 2
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx
        
        if closest_idx >= 0:
            component_colors[closest_idx] = color_val
    
    # Build output
    if use_horizontal:
        # Place components side-by-side (horizontally concatenated)
        result = [[] for _ in range(4)]
        
        for idx, (bin_grid, _, _, _) in enumerate(component_grids):
            color = component_colors[idx]
            for row_idx, row in enumerate(bin_grid):
                result_row = [color if cell == 1 else 0 for cell in row]
                result[row_idx].extend(result_row)
    else:
        # Stack components vertically
        result = []
        
        for idx, (bin_grid, _, _, _) in enumerate(component_grids):
            color = component_colors[idx]
            for row in bin_grid:
                result_row = [color if cell == 1 else 0 for cell in row]
                result.append(result_row)
    
    return result


def main():
    # Load task JSON
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/50aad11f.json'
    
    with open(json_path) as f:
        task = json.load(f)
    
    # Test on training examples
    all_pass = True
    for idx, example in enumerate(task['train']):
        output = solve(example['input'])
        expected = example['output']
        
        passed = output == expected
        all_pass = all_pass and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"Training Example {idx + 1}: {status}")
        if not passed:
            print(f"  Expected shape: {len(expected)}x{len(expected[0]) if expected else 0}")
            print(f"  Got shape: {len(output)}x{len(output[0]) if output else 0}")
            print(f"  Expected: {expected}")
            print(f"  Got: {output}")
    
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
