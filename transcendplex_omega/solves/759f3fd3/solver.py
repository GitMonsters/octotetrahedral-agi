import json

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    ARC puzzle 759f3fd3: Color swap with checkerboard pattern.
    
    Rule: A cross of 3s divides the grid. For each non-3 cell:
    - If max(distance_to_cross_row, distance_to_cross_col) is EVEN -> fill with 4
    - Otherwise -> keep as 0
    """
    h = len(grid)
    w = len(grid[0])
    
    # Find the cross (horizontal and vertical line of 3s)
    cross_row = -1
    cross_col = -1
    
    for i in range(h):
        if all(grid[i][j] == 3 for j in range(w)):
            cross_row = i
            break
    
    for j in range(w):
        if all(grid[i][j] == 3 for i in range(h)):
            cross_col = j
            break
    
    result = [row[:] for row in grid]
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 3:
                result[i][j] = 3
            else:
                dr = abs(i - cross_row)
                dc = abs(j - cross_col)
                max_dist = max(dr, dc)
                result[i][j] = 4 if max_dist % 2 == 0 else 0
    
    return result


def test():
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/759f3fd3.json', 'r') as f:
        data = json.load(f)
    
    all_passed = True
    for idx, example in enumerate(data['train']):
        input_grid = example['input']
        expected = example['output']
        predicted = solve(input_grid)
        
        if predicted == expected:
            print(f"✓ Training example {idx + 1} PASSED")
        else:
            print(f"✗ Training example {idx + 1} FAILED")
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TRAINING EXAMPLES PASSED!")


if __name__ == '__main__':
    test()
