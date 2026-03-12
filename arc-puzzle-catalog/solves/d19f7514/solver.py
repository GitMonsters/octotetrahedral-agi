def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Extract and merge two 6x4 regions of a 12x4 grid.
    
    Input: 12x4 grid
      - Top half (rows 0-5): Contains 0 and 3
      - Bottom half (rows 6-11): Contains 0 and 5
    
    Output: 6x4 grid
      - For each position (i,j): output is 4 if top[i][j]==3 OR bottom[i][j]==5, else 0
    """
    top = grid[:6]
    bottom = grid[6:12]
    
    output = []
    for i in range(6):
        row = []
        for j in range(4):
            has_3 = top[i][j] == 3
            has_5 = bottom[i][j] == 5
            value = 4 if (has_3 or has_5) else 0
            row.append(value)
        output.append(row)
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/d19f7514.json') as f:
        data = json.load(f)
    
    print("Testing on training examples...")
    all_passed = True
    for idx, example in enumerate(data['train']):
        result = solve(example['input'])
        expected = example['output']
        if result == expected:
            print(f"  Example {idx+1}: ✓ PASS")
        else:
            print(f"  Example {idx+1}: ✗ FAIL")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False
    
    print("\nTesting on test example...")
    result = solve(data['test'][0]['input'])
    expected = data['test'][0]['output']
    if result == expected:
        print(f"  Test: ✓ PASS")
    else:
        print(f"  Test: ✗ FAIL")
        print(f"    Expected: {expected}")
        print(f"    Got:      {result}")
        all_passed = False
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
