"""
ARC Puzzle 77d40a18 Solver

Pattern: 
- Find 2x2 squares of foreground color → convert to gray (5)
- All other foreground cells → convert to magenta (6)
"""

from collections import Counter


def transform(grid):
    grid = [list(row) for row in grid]
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background (most common) and foreground colors
    color_counts = Counter(c for row in grid for c in row)
    background = color_counts.most_common(1)[0][0]
    foreground = None
    for color, _ in color_counts.most_common():
        if color != background:
            foreground = color
            break
    
    if foreground is None:
        return grid
    
    # Find all positions that are part of a 2x2 square of foreground color
    part_of_2x2 = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (grid[r][c] == foreground and
                grid[r+1][c] == foreground and
                grid[r][c+1] == foreground and
                grid[r+1][c+1] == foreground):
                part_of_2x2.add((r, c))
                part_of_2x2.add((r+1, c))
                part_of_2x2.add((r, c+1))
                part_of_2x2.add((r+1, c+1))
    
    # Create output: 2x2 parts → gray(5), other foreground → magenta(6)
    output = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == foreground:
                output[r][c] = 5 if (r, c) in part_of_2x2 else 6
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['77d40a18']
    
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        print(f"Example {i}: {'✓ PASS' if match else '✗ FAIL'}")
        
        if not match:
            print("Expected:")
            for row in expected:
                print(row)
            print("Got:")
            for row in result:
                print(row)
