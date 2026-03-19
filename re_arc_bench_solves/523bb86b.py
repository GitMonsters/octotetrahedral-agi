"""
ARC Puzzle 523bb86b Solver

Pattern: From each non-background marker point, draw diagonal lines in all 4 
directions (NE, NW, SE, SW). The marker stays its original color, the diagonal
cells become magenta (6).
"""

def transform(grid):
    import copy
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg_color = Counter(flat).most_common(1)[0][0]
    
    # Find all marker positions (non-background)
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg_color:
                markers.append((r, c, grid[r][c]))
    
    # Create output grid
    output = copy.deepcopy(grid)
    
    # From each marker, draw diagonals in all 4 directions
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # NW, NE, SW, SE
    
    for mr, mc, marker_color in markers:
        for dr, dc in directions:
            r, c = mr + dr, mc + dc
            while 0 <= r < rows and 0 <= c < cols:
                if output[r][c] == bg_color:  # Only fill background cells
                    output[r][c] = 6  # Magenta
                r += dr
                c += dc
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['523bb86b']
    
    print("Testing on all training examples:")
    all_pass = True
    for i, ex in enumerate(task['train']):
        result = transform(ex['input'])
        expected = ex['output']
        match = result == expected
        all_pass = all_pass and match
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
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
