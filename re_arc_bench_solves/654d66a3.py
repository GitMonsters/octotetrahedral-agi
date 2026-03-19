"""
Solver for ARC puzzle 654d66a3

Pattern: Find rectangles defined by 4 corner markers of the same non-background color.
Complete the pattern inside each rectangle to make it point-symmetric (180° rotational symmetry).
"""

def transform(grid):
    import copy
    grid = [list(row) for row in grid]
    rows, cols = len(grid), len(grid[0])
    
    # Find background color (most common)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find all non-background points by color
    color_points = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                color = grid[r][c]
                if color not in color_points:
                    color_points[color] = []
                color_points[color].append((r, c))
    
    output = copy.deepcopy(grid)
    
    # For each color, find all rectangles with 4 corners of that color
    for color, points in color_points.items():
        if len(points) < 4:
            continue
        
        # Find all valid rectangles (4 corners of same color)
        rectangles = []
        points_set = set(points)
        
        for i, (r1, c1) in enumerate(points):
            for j, (r2, c2) in enumerate(points):
                if j <= i:
                    continue
                if r1 >= r2 or c1 >= c2:
                    continue
                # Check if (r1, c2) and (r2, c1) exist
                if (r1, c2) in points_set and (r2, c1) in points_set:
                    rectangles.append((r1, c1, r2, c2))
        
        # For each rectangle, make the interior point-symmetric
        for (r1, c1, r2, c2) in rectangles:
            # The rectangle spans from (r1, c1) to (r2, c2)
            # Center is at ((r1+r2)/2, (c1+c2)/2)
            # For point symmetry, point (r, c) maps to (r1+r2-r, c1+c2-c)
            
            # Collect all non-background points in/around the rectangle
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    # Map to symmetric point
                    sr = r1 + r2 - r
                    sc = c1 + c2 - c
                    
                    # If current point has non-bg and symmetric is bg, copy
                    if grid[r][c] != bg and output[sr][sc] == bg:
                        output[sr][sc] = grid[r][c]
                    # If symmetric has non-bg and current is bg, copy
                    if grid[sr][sc] != bg and output[r][c] == bg:
                        output[r][c] = grid[sr][sc]
    
    return [list(row) for row in output]


if __name__ == "__main__":
    import json
    
    # Load task
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['654d66a3']
    
    print("Testing on all training examples...")
    all_pass = True
    
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        all_pass = all_pass and match
        
        print(f"\nExample {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            print("Input:")
            for row in inp:
                print(row)
            print("Expected:")
            for row in expected:
                print(row)
            print("Got:")
            for row in result:
                print(row)
            # Show differences
            print("Differences:")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): expected {expected[r][c]}, got {result[r][c]}")
    
    print(f"\n{'='*50}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
