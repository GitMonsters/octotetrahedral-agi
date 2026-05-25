def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    The input is a 15x5 grid composed of 3 stacked 5x5 layers:
    - Layer 0 (rows 0-4): contains color 1
    - Layer 1 (rows 5-9): contains color 8
    - Layer 2 (rows 10-14): contains color 6
    
    The output is a 5x5 grid where each cell contains the color from the
    highest priority color at that position across all three layers.
    
    Priority: 6 > 1 > 8 > 0
    """
    layer0 = grid[0:5]
    layer1 = grid[5:10]
    layer2 = grid[10:15]
    
    result = []
    for i in range(5):
        row = []
        for j in range(5):
            colors = [layer0[i][j], layer1[i][j], layer2[i][j]]
            
            # Priority: 6 > 1 > 8 > 0
            if 6 in colors:
                row.append(6)
            elif 1 in colors:
                row.append(1)
            elif 8 in colors:
                row.append(8)
            else:
                row.append(0)
        result.append(row)
    
    return result


if __name__ == "__main__":
    import json
    
    # Load test data
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/6a11f6da.json') as f:
        data = json.load(f)
    
    # Test all training examples
    all_pass = True
    for ex_idx, train_ex in enumerate(data['train']):
        inp = train_ex['input']
        expected = train_ex['output']
        result = solve(inp)
        
        if result == expected:
            print(f"✓ Training example {ex_idx} PASSED")
        else:
            print(f"✗ Training example {ex_idx} FAILED")
            all_pass = False
            # Show first difference
            for i in range(5):
                for j in range(5):
                    if result[i][j] != expected[i][j]:
                        print(f"  First diff at [{i},{j}]: got {result[i][j]}, expected {expected[i][j]}")
                        break
                if result[i] != expected[i]:
                    break
    
    if all_pass:
        print("\n✓ ALL TRAINING EXAMPLES PASSED")
    else:
        print("\n✗ SOME EXAMPLES FAILED")
