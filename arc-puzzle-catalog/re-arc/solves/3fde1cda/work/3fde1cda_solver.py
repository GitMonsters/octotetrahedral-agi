#!/usr/bin/env python3
"""Solver for ARC task 3fde1cda."""

def transform(grid):
    """
    Transform the grid by finding 4 corner markers and cropping to that region.
    
    The corner markers are isolated cells of the same color that:
    - Are single cells (not part of a larger block)
    - Form exactly 4 corners of a rectangle
    - Are the only 4 isolated cells of that color
    """
    grid = [list(row) for row in grid]
    h, w = len(grid), len(grid[0])
    
    # Find background color (most common)
    color_counts = {}
    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1
    background = max(color_counts.items(), key=lambda x: x[1])[0]
    
    # Find all isolated cells (surrounded by background or grid edge)
    # An isolated cell is one where all 4 orthogonal neighbors are background
    isolated_cells = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != background:
                val = grid[r][c]
                
                # Check if orthogonally isolated (all 4 neighbors are background or edge)
                is_isolated = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if grid[nr][nc] == val:
                            is_isolated = False
                            break
                
                if is_isolated:
                    if val not in isolated_cells:
                        isolated_cells[val] = []
                    isolated_cells[val].append((r, c))
    
    # Find the color that has exactly 4 isolated cells forming a perfect rectangle
    corner_markers = None
    corner_color = None
    
    for color, positions in isolated_cells.items():
        if len(positions) == 4:
            # Check if they form a perfect rectangle
            rows = sorted(set(r for r, c in positions))
            cols = sorted(set(c for r, c in positions))
            
            if len(rows) == 2 and len(cols) == 2:
                # Verify all 4 corners exist
                expected = [
                    (rows[0], cols[0]), (rows[0], cols[1]),
                    (rows[1], cols[0]), (rows[1], cols[1])
                ]
                if sorted(positions) == sorted(expected):
                    corner_markers = (rows[0], rows[1], cols[0], cols[1])
                    corner_color = color
                    break
    
    # If we didn't find exactly 4, try to find 4 cells that form a rectangle
    if not corner_markers:
        for color in sorted(isolated_cells.keys(), key=lambda c: len(isolated_cells[c]), reverse=True):
            positions = isolated_cells[color]
            if len(positions) >= 4:
                # Try all combinations of 4 positions to find a valid rectangle
                from itertools import combinations
                for combo in combinations(positions, 4):
                    rows = sorted(set(r for r, c in combo))
                    cols = sorted(set(c for r, c in combo))
                    
                    if len(rows) == 2 and len(cols) == 2:
                        # Check if this is a valid rectangle
                        expected = [
                            (rows[0], cols[0]), (rows[0], cols[1]),
                            (rows[1], cols[0]), (rows[1], cols[1])
                        ]
                        if sorted(combo) == sorted(expected):
                            corner_markers = (rows[0], rows[1], cols[0], cols[1])
                            corner_color = color
                            break
                
                if corner_markers:
                    break
    
    if not corner_markers:
        # No valid corners found, return original
        return grid
    
    r1, r2, c1, c2 = corner_markers
    
    # Crop to the region defined by corners (inclusive)
    cropped = []
    for r in range(r1, r2 + 1):
        cropped.append(grid[r][c1:c2 + 1])
    
    return cropped


if __name__ == '__main__':
    import json
    
    # Load the task
    with open('/tmp/rearc45/3fde1cda.json', 'r') as f:
        data = json.load(f)
    
    # Test on training examples
    print("Testing on training examples:")
    all_pass = True
    
    for i, pair in enumerate(data['train']):
        input_grid = pair['input']
        expected_output = pair['output']
        predicted_output = transform(input_grid)
        
        match = predicted_output == expected_output
        all_pass = all_pass and match
        
        print(f"\nTraining pair {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            print(f"Expected shape: {len(expected_output)}x{len(expected_output[0])}")
            print(f"Got shape: {len(predicted_output)}x{len(predicted_output[0])}")
            
            # Show first few rows of diff
            print("\nExpected (first 5 rows):")
            for row in expected_output[:5]:
                print(row)
            print("\nGot (first 5 rows):")
            for row in predicted_output[:5]:
                print(row)
    
    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*60}")
