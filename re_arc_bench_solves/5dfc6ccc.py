"""
ARC Puzzle 5dfc6ccc

Pattern: Colored pixels on edges create 3-wide stripes with diagonal checkerboard pattern.
- Pixels on top/bottom row → vertical stripes
- Pixels on left/right column → horizontal stripes
- Checkerboard uses (row + col) % 2 parity matching the original pixel
"""

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Copy grid
    output = [row[:] for row in grid]
    
    # Find non-background pixels
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                markers.append((r, c, grid[r][c]))
    
    # Process each marker
    for r, c, color in markers:
        parity = (r + c) % 2
        
        # Determine if on top/bottom (vertical stripe) or left/right (horizontal stripe)
        on_top_bottom = (r == 0 or r == rows - 1)
        on_left_right = (c == 0 or c == cols - 1)
        
        if on_top_bottom:
            # Vertical stripe from c-1 to c+1, all rows
            for rr in range(rows):
                for cc in range(max(0, c-1), min(cols, c+2)):
                    if (rr + cc) % 2 == parity:
                        output[rr][cc] = color
        elif on_left_right:
            # Horizontal stripe from r-1 to r+1, all columns
            for rr in range(max(0, r-1), min(rows, r+2)):
                for cc in range(cols):
                    if (rr + cc) % 2 == parity:
                        output[rr][cc] = color
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['5dfc6ccc']
    
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        print(f"Example {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            print("Expected:")
            for row in expected:
                print(row)
            print("Got:")
            for row in result:
                print(row)
            print()
