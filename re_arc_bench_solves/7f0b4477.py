"""
ARC Puzzle 7f0b4477 Solver

Pattern: From each non-background cell, draw diagonal rays in all 4 diagonal 
directions using that cell's color. Rays extend to grid boundaries.
"""

def transform(grid):
    import copy
    from collections import Counter
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the background color (most common)
    flat = [c for row in grid for c in row]
    bg_color = Counter(flat).most_common(1)[0][0]
    
    # Create output grid (copy of input)
    output = copy.deepcopy(grid)
    
    # Find all non-background cells (markers)
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg_color:
                markers.append((r, c, grid[r][c]))
    
    # Four diagonal directions: NW, NE, SW, SE
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # For each marker, draw diagonal rays in all 4 directions
    for r, c, color in markers:
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            while 0 <= nr < rows and 0 <= nc < cols:
                output[nr][nc] = color
                nr += dr
                nc += dc
    
    return output


if __name__ == "__main__":
    import json
    
    # Load the task
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['7f0b4477']
    
    # Test on all training examples
    all_pass = True
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        all_pass = all_pass and match
        
        print(f"Training Example {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            # Show first difference
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  First diff at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
                        break
                else:
                    continue
                break
    
    print(f"\nAll training examples pass: {all_pass}")
