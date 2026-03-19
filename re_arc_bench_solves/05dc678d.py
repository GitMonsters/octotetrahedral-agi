"""
ARC Puzzle 05dc678d Solver

Pattern: Find rectangular regions filled entirely with 3s (green),
then draw a border/frame of 7s (orange) around each rectangle.
"""

def transform(grid):
    import copy
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)
    
    # Find all rectangles filled with 3s
    visited = [[False]*cols for _ in range(rows)]
    rectangles = []
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3 and not visited[r][c]:
                # Try to find the largest rectangle of 3s starting here
                # Find extent of potential rectangle
                max_c = c
                while max_c < cols and grid[r][max_c] == 3:
                    max_c += 1
                max_c -= 1
                
                # Now find how far down we can go with this width
                max_r = r
                valid = True
                while max_r < rows and valid:
                    for cc in range(c, max_c + 1):
                        if grid[max_r][cc] != 3:
                            valid = False
                            break
                    if valid:
                        max_r += 1
                max_r -= 1
                
                # Check if this is a valid rectangle (at least 2x2)
                height = max_r - r + 1
                width = max_c - c + 1
                
                if height >= 2 and width >= 2:
                    # Verify it's a complete rectangle of 3s
                    is_rect = True
                    for rr in range(r, max_r + 1):
                        for cc in range(c, max_c + 1):
                            if grid[rr][cc] != 3:
                                is_rect = False
                                break
                        if not is_rect:
                            break
                    
                    if is_rect:
                        rectangles.append((r, c, max_r, max_c))
                        # Mark as visited
                        for rr in range(r, max_r + 1):
                            for cc in range(c, max_c + 1):
                                visited[rr][cc] = True
    
    # Draw border of 7s around each rectangle
    for (r1, c1, r2, c2) in rectangles:
        # Top border (row above)
        if r1 > 0:
            for cc in range(c1, c2 + 1):
                result[r1 - 1][cc] = 7
        # Bottom border (row below)
        if r2 < rows - 1:
            for cc in range(c1, c2 + 1):
                result[r2 + 1][cc] = 7
        # Left border (column left)
        if c1 > 0:
            for rr in range(r1, r2 + 1):
                result[rr][c1 - 1] = 7
        # Right border (column right)
        if c2 < cols - 1:
            for rr in range(r1, r2 + 1):
                result[rr][c2 + 1] = 7
        # Corners
        if r1 > 0 and c1 > 0:
            result[r1 - 1][c1 - 1] = 7
        if r1 > 0 and c2 < cols - 1:
            result[r1 - 1][c2 + 1] = 7
        if r2 < rows - 1 and c1 > 0:
            result[r2 + 1][c1 - 1] = 7
        if r2 < rows - 1 and c2 < cols - 1:
            result[r2 + 1][c2 + 1] = 7
    
    return result


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    puzzle = data['05dc678d']
    
    all_pass = True
    for i, ex in enumerate(puzzle['train']):
        inp = ex['input']
        expected = ex['output']
        actual = transform(inp)
        
        if actual == expected:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            all_pass = False
            # Show differences
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if actual[r][c] != expected[r][c]:
                        print(f"  Diff at ({r},{c}): expected {expected[r][c]}, got {actual[r][c]}")
    
    print(f"\nAll pass: {all_pass}")
