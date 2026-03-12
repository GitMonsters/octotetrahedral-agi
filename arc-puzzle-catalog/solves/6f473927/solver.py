def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Rule: Double the width based on first element of first row
    If grid[0][0] == 0:
      - Left: original, Right: flip + invert
    If grid[0][0] == 2:
      - Left: flip + invert, Right: original
    """
    if not grid or not grid[0]:
        return grid
    
    use_flip_left = grid[0][0] == 2
    result = []
    
    for row in grid:
        flipped = row[::-1]
        inverted = [8 if cell == 0 else 0 for cell in flipped]
        
        if use_flip_left:
            # flip+invert on left, original on right
            new_row = inverted + row
        else:
            # original on left, flip+invert on right
            new_row = row + inverted
        
        result.append(new_row)
    
    return result


if __name__ == "__main__":
    import json
    
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/6f473927.json") as f:
        data = json.load(f)
    
    all_pass = True
    for i, example in enumerate(data["train"]):
        result = solve(example["input"])
        expected = example["output"]
        
        if result == expected:
            print(f"✓ Training example {i+1} PASSED")
        else:
            print(f"✗ Training example {i+1} FAILED")
            print(f"  Expected shape: {len(expected)}x{len(expected[0]) if expected else 0}")
            print(f"  Got shape: {len(result)}x{len(result[0]) if result else 0}")
            all_pass = False
    
    if all_pass:
        print("\n✓ ALL TRAINING EXAMPLES PASSED")
    else:
        print("\n✗ SOME EXAMPLES FAILED")
