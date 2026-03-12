import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solve e345f17b: Split 4x8 grid into left (cols 0-3) and right (cols 4-7).
    Output 4x4 grid where each cell is 4 if both corresponding left and right cells are 0, else 0.
    """
    result = []
    for r in range(4):
        row = []
        for c in range(4):
            left_val = grid[r][c]
            right_val = grid[r][c + 4]
            
            if left_val == 0 and right_val == 0:
                row.append(4)
            else:
                row.append(0)
        result.append(row)
    
    return result


if __name__ == '__main__':
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/e345f17b.json') as f:
        data = json.load(f)
    
    print("Testing on training examples:")
    all_pass = True
    for idx, example in enumerate(data['train']):
        inp = example['input']
        expected = example['output']
        predicted = solve(inp)
        
        match = predicted == expected
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"  Example {idx + 1}: {status}")
        
        if not match:
            all_pass = False
            print(f"    Expected: {expected}")
            print(f"    Got:      {predicted}")
    
    print(f"\n{'All tests PASSED!' if all_pass else 'Some tests FAILED!'}")
    
    print("\nTesting on test examples:")
    for idx, example in enumerate(data['test']):
        inp = example['input']
        expected = example['output']
        predicted = solve(inp)
        
        match = predicted == expected
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"  Test {idx + 1}: {status}")
        
        if not match:
            print(f"    Expected: {expected}")
            print(f"    Got:      {predicted}")
