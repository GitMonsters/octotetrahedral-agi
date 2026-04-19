"""
ARC Puzzle 18b4093c Solver

Rule: Find 2x2 blocks of specific colors and draw diagonals:
- Color 6 blocks: diagonal goes DOWN-RIGHT from bottom-right corner
- Color 7 blocks: diagonal goes UP-LEFT from top-left corner
"""

def transform(grid):
    import copy
    grid = [list(row) for row in grid]
    output = copy.deepcopy(grid)
    rows = len(grid)
    cols = len(grid[0])
    
    # Find 2x2 blocks
    blocks = []  # (top_row, left_col, color)
    
    for r in range(rows - 1):
        for c in range(cols - 1):
            color = grid[r][c]
            # Check if this is top-left of a 2x2 block
            if (grid[r][c] == color and 
                grid[r][c+1] == color and 
                grid[r+1][c] == color and 
                grid[r+1][c+1] == color):
                # Check it's not background (need to detect background)
                # Count colors to find background
                pass
    
    # Detect background (most frequent color)
    from collections import Counter
    all_colors = [grid[r][c] for r in range(rows) for c in range(cols)]
    background = Counter(all_colors).most_common(1)[0][0]
    
    # Find all 2x2 blocks of non-background colors
    visited = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            color = grid[r][c]
            if color == background:
                continue
            if (r, c) in visited:
                continue
            # Check if this is top-left of a 2x2 block
            if (grid[r][c] == color and 
                grid[r][c+1] == color and 
                grid[r+1][c] == color and 
                grid[r+1][c+1] == color):
                blocks.append((r, c, color))
                visited.add((r, c))
                visited.add((r, c+1))
                visited.add((r+1, c))
                visited.add((r+1, c+1))
    
    # Draw diagonals for each block
    for r, c, color in blocks:
        if color == 6:
            # Diagonal goes DOWN-RIGHT from bottom-right corner (r+1, c+1)
            dr, dc = r + 2, c + 2
            while dr < rows and dc < cols:
                output[dr][dc] = color
                dr += 1
                dc += 1
        elif color == 7:
            # Diagonal goes UP-LEFT from top-left corner (r, c)
            dr, dc = r - 1, c - 1
            while dr >= 0 and dc >= 0:
                output[dr][dc] = color
                dr -= 1
                dc -= 1
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['18b4093c']
    
    all_passed = True
    for i, ex in enumerate(task['train']):
        input_grid = ex['input']
        expected = ex['output']
        result = transform(input_grid)
        
        passed = result == expected
        all_passed = all_passed and passed
        
        print(f"Training Example {i}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print("Expected:")
            for row in expected[:5]:
                print(row)
            print("Got:")
            for row in result[:5]:
                print(row)
            print()
    
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
