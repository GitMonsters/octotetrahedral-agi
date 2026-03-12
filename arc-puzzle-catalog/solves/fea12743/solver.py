#!/usr/bin/env python3
"""
Solver for ARC puzzle fea12743
Rule: 
1. Find quad with highest count of 2s -> color 3
2. For other quads, apply checkerboard pattern based on grid position
"""

def extract_quadrant_pattern(grid, r_start, r_end, c_start, c_end):
    """Extract pattern from a quadrant"""
    pattern = []
    for r in range(r_start, r_end):
        row = []
        for c in range(c_start, c_end):
            row.append(grid[r][c])
        pattern.append(row)
    return tuple(tuple(r) for r in pattern)

def count_2s(pattern):
    """Count number of 2s in pattern"""
    count = 0
    for row in pattern:
        for cell in row:
            if cell == 2:
                count += 1
    return count

def solve(grid):
    """Apply the transformation rule"""
    # Quadrant coordinates and positions in 3x2 grid
    quad_coords = {
        "TL": (1, 5, 1, 5),
        "TR": (1, 5, 6, 10),
        "ML": (6, 10, 1, 5),
        "MR": (6, 10, 6, 10),
        "BL": (11, 15, 1, 5),
        "BR": (11, 15, 6, 10),
    }
    
    quad_positions = {
        "TL": (0, 0),
        "TR": (0, 1),
        "ML": (1, 0),
        "MR": (1, 1),
        "BL": (2, 0),
        "BR": (2, 1),
    }
    
    quad_order = ["TL", "TR", "ML", "MR", "BL", "BR"]
    
    # Extract all quadrants and their counts
    quads = {}
    counts = {}
    for name, (r_start, r_end, c_start, c_end) in quad_coords.items():
        pattern = extract_quadrant_pattern(grid, r_start, r_end, c_start, c_end)
        count = count_2s(pattern)
        quads[name] = (pattern, count)
        counts[name] = count
    
    # Find max count quad
    max_count_quad = max(counts, key=counts.get)
    
    # Find the first non-3 quad to determine checkerboard orientation
    # We'll use the position of the first non-max-count quad
    non_max_quads = [n for n in quad_order if n != max_count_quad]
    if non_max_quads:
        first_non_max = non_max_quads[0]
        # For now, use orientation 1: (row+col)%2 -> 0=2, 1=8
        orientation = 1
    
    # Create output
    output = [row[:] for row in grid]
    
    # Apply colors
    for name in quad_order:
        r_start, r_end, c_start, c_end = quad_coords[name]
        row_pos, col_pos = quad_positions[name]
        
        if name == max_count_quad:
            color = 3
        else:
            # Checkerboard pattern (orientation 1)
            checkerboard_val = (row_pos + col_pos) % 2
            color = 2 if checkerboard_val == 0 else 8
        
        # Apply color to all non-zero cells in this quadrant
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                if grid[r][c] != 0:
                    output[r][c] = color
    
    return output


if __name__ == "__main__":
    import json
    import sys
    
    # Test on training examples
    task_file = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/fea12743.json"
    task = json.load(open(task_file))
    
    passed = 0
    failed = 0
    
    for idx, example in enumerate(task['train']):
        result = solve(example['input'])
        if result == example['output']:
            print(f"✓ Train {idx} PASS")
            passed += 1
        else:
            print(f"✗ Train {idx} FAIL")
            # Find first difference
            for r in range(len(example['input'])):
                for c in range(len(example['input'][0])):
                    if result[r][c] != example['output'][r][c]:
                        print(f"  First diff at ({r},{c}): got {result[r][c]}, expected {example['output'][r][c]}")
                        break
                else:
                    continue
                break
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
