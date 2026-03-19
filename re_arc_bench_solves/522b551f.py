"""
ARC Puzzle 522b551f Solver

Pattern: A small corner template of colored points defines nested rectangular frames.
The output reflects this pattern to create full horizontal and vertical symmetry.
Each color's row/column position determines its frame level.
"""

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find all colored points (non-background)
    points = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                points.append((r, c, grid[r][c]))
    
    if not points:
        return [row[:] for row in grid]
    
    # Find bounding box of colored points
    min_r = min(p[0] for p in points)
    max_r = max(p[0] for p in points)
    min_c = min(p[1] for p in points)
    max_c = max(p[1] for p in points)
    
    # Determine which corner the pattern is in
    center_r = rows // 2
    center_c = cols // 2
    avg_r = (min_r + max_r) / 2
    avg_c = (min_c + max_c) / 2
    
    # Create output grid
    output = [[bg for _ in range(cols)] for _ in range(rows)]
    
    # Normalize points to corner (0, 0) relative coordinates
    # Based on which quadrant the pattern is in
    normalized = []
    for r, c, color in points:
        # Calculate relative position from closest corner
        rel_r = r - min_r
        rel_c = c - min_c
        normalized.append((rel_r, rel_c, color))
    
    # Determine the step size (distance between colored points, usually 2)
    r_vals = sorted(set(p[0] for p in normalized))
    c_vals = sorted(set(p[1] for p in normalized))
    
    step_r = 2
    step_c = 2
    if len(r_vals) > 1:
        diffs = [r_vals[i+1] - r_vals[i] for i in range(len(r_vals)-1)]
        step_r = min(diffs) if diffs else 2
    if len(c_vals) > 1:
        diffs = [c_vals[i+1] - c_vals[i] for i in range(len(c_vals)-1)]
        step_c = min(diffs) if diffs else 2
    
    # Build a map of (level_r, level_c) -> color from normalized points
    level_map = {}
    for rel_r, rel_c, color in normalized:
        lvl_r = rel_r // step_r
        lvl_c = rel_c // step_c
        level_map[(lvl_r, lvl_c)] = color
    
    # Find max levels
    max_lvl_r = max(p[0] for p in level_map.keys())
    max_lvl_c = max(p[1] for p in level_map.keys())
    
    # For each odd position in output, determine the color
    # based on distance from edges (creating symmetric frames)
    for r in range(rows):
        for c in range(cols):
            # Only fill odd positions
            if r % step_r != 1 or c % step_c != 1:
                continue
            
            # Calculate level from edge (0 = outermost)
            # Level is distance from nearest edge in "step" units
            dist_top = (r - 1) // step_r
            dist_bot = (rows - 1 - r - 1) // step_r
            dist_left = (c - 1) // step_c
            dist_right = (cols - 1 - c - 1) // step_c
            
            lvl_r = min(dist_top, dist_bot)
            lvl_c = min(dist_left, dist_right)
            
            # Clamp to max levels
            lvl_r = min(lvl_r, max_lvl_r)
            lvl_c = min(lvl_c, max_lvl_c)
            
            # Find the color for this level combination
            if (lvl_r, lvl_c) in level_map:
                output[r][c] = level_map[(lvl_r, lvl_c)]
            else:
                # Fill with the innermost defined color at this row level
                # Try to find closest match
                for try_c in range(lvl_c, -1, -1):
                    if (lvl_r, try_c) in level_map:
                        output[r][c] = level_map[(lvl_r, try_c)]
                        break
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['522b551f']
    
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        print(f"Example {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            print(f"  Input size: {len(inp)}x{len(inp[0])}")
            print(f"  Output size: {len(result)}x{len(result[0])}")
            print(f"  Expected size: {len(expected)}x{len(expected[0])}")
            # Show differences
            for r in range(min(len(result), len(expected))):
                for c in range(min(len(result[0]), len(expected[0]))):
                    if result[r][c] != expected[r][c]:
                        print(f"    Diff at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
