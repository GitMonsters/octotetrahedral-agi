"""
ARC Puzzle 0abd034e Solver

Pattern: Find rows and columns that are entirely filled with color 6 (magenta).
Fill the intersection of these all-6 rows and all-6 columns with color 5 (gray).
"""

def transform(grid):
    import copy
    grid = [list(row) for row in grid]
    h = len(grid)
    w = len(grid[0])
    
    # Find rows that are entirely 6
    all_6_rows = set()
    for r in range(h):
        if all(grid[r][c] == 6 for c in range(w)):
            all_6_rows.add(r)
    
    # Find columns that are entirely 6
    all_6_cols = set()
    for c in range(w):
        if all(grid[r][c] == 6 for r in range(h)):
            all_6_cols.add(c)
    
    # Create output grid
    output = copy.deepcopy(grid)
    
    # Fill intersections of all-6 rows and all-6 cols with 5
    for r in all_6_rows:
        for c in range(w):
            output[r][c] = 5
    
    for c in all_6_cols:
        for r in range(h):
            output[r][c] = 5
    
    return output


if __name__ == "__main__":
    import json
    
    # Load task data
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['0abd034e']
    
    # Test on all training examples
    all_passed = True
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        passed = result == expected
        all_passed = all_passed and passed
        
        print(f"Train {i}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"  Expected output shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got output shape: {len(result)}x{len(result[0])}")
            # Find first difference
            for r in range(min(len(expected), len(result))):
                for c in range(min(len(expected[0]), len(result[0]))):
                    if expected[r][c] != result[r][c]:
                        print(f"  First diff at ({r},{c}): expected {expected[r][c]}, got {result[r][c]}")
                        break
                else:
                    continue
                break
    
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
