"""
ARC Puzzle 48487fa9 Solver

Pattern: Point reflection/central symmetry
- Find all non-background (foreground) pixels
- Calculate the center of the bounding box of foreground pixels
- Reflect each foreground pixel through that center
- Mark reflected positions with color 9 if they're currently background
"""

def transform(grid):
    import copy
    grid = [list(row) for row in grid]
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] = color_counts.get(grid[r][c], 0) + 1
    background = max(color_counts, key=color_counts.get)
    
    # Find all foreground pixels
    foreground_pixels = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                foreground_pixels.append((r, c))
    
    if not foreground_pixels:
        return grid
    
    # Find bounding box center
    min_r = min(p[0] for p in foreground_pixels)
    max_r = max(p[0] for p in foreground_pixels)
    min_c = min(p[1] for p in foreground_pixels)
    max_c = max(p[1] for p in foreground_pixels)
    
    center_r = (min_r + max_r) / 2.0
    center_c = (min_c + max_c) / 2.0
    
    # Reflect each foreground pixel through center and mark with 9
    result = copy.deepcopy(grid)
    for r, c in foreground_pixels:
        # Reflect through center
        new_r = int(2 * center_r - r)
        new_c = int(2 * center_c - c)
        
        # If in bounds and currently background, mark with 9
        if 0 <= new_r < rows and 0 <= new_c < cols:
            if result[new_r][new_c] == background:
                result[new_r][new_c] = 9
    
    return result


if __name__ == "__main__":
    import json
    
    # Load task
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['48487fa9']
    
    print("Testing on all training examples:")
    all_pass = True
    
    for i, ex in enumerate(task['train']):
        input_grid = ex['input']
        expected = ex['output']
        result = transform(input_grid)
        
        match = result == expected
        all_pass = all_pass and match
        print(f"\nExample {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            print("Expected:")
            for row in expected:
                print(row)
            print("Got:")
            for row in result:
                print(row)
            # Find differences
            print("Differences:")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): expected {expected[r][c]}, got {result[r][c]}")
    
    print(f"\n{'='*50}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
