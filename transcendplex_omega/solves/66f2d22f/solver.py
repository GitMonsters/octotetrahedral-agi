def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    ARC-AGI Puzzle 66f2d22f Solver
    
    Rule: Split the 4x14 grid into left (cols 0-6) and right (cols 7-13) halves.
    For each cell, output 5 if (left != 3 AND right != 2), else output 0.
    Result is a 4x7 grid.
    """
    rows = len(grid)
    cols = 7
    result = []
    
    for row in grid:
        output_row = []
        for j in range(cols):
            left_val = row[j]
            right_val = row[j + cols]
            
            output_val = 5 if (left_val != 3 and right_val != 2) else 0
            output_row.append(output_val)
        result.append(output_row)
    
    return result


if __name__ == "__main__":
    import json
    
    # Load puzzle data
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/66f2d22f.json') as f:
        data = json.load(f)
    
    print("Testing ARC-AGI Puzzle 66f2d22f Solver")
    print("=" * 50)
    
    # Test on all training examples
    all_pass = True
    for idx, example in enumerate(data['train']):
        predicted = solve(example['input'])
        expected = example['output']
        
        match = predicted == expected
        all_pass = all_pass and match
        
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"Training Example {idx+1}: {status}")
        
        if not match:
            print(f"  Predicted: {predicted}")
            print(f"  Expected:  {expected}")
    
    print("=" * 50)
    print(f"Result: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
