import json

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    ARC puzzle 8597cfd7 solver.
    
    Rule: Find the row with all 5s (divider).
    Count occurrences of colors 2 and 4 in top and bottom sections.
    Return the color that increases MORE from top to bottom section.
    """
    # Find the divider row (all 5s)
    divider_row = -1
    for i, row in enumerate(grid):
        if all(cell == 5 for cell in row):
            divider_row = i
            break
    
    if divider_row == -1:
        return [[0, 0], [0, 0]]
    
    # Count colors in top and bottom sections
    count_2_top = sum(row.count(2) for row in grid[:divider_row])
    count_4_top = sum(row.count(4) for row in grid[:divider_row])
    count_2_bot = sum(row.count(2) for row in grid[divider_row + 1:])
    count_4_bot = sum(row.count(4) for row in grid[divider_row + 1:])
    
    # Calculate the increase from top to bottom for each color
    increase_2 = count_2_bot - count_2_top
    increase_4 = count_4_bot - count_4_top
    
    # Pick the color with the larger increase (or 2 as default if tied)
    winning_color = 2 if increase_2 >= increase_4 else 4
    
    # Return 2x2 grid filled with winning color
    return [[winning_color, winning_color], [winning_color, winning_color]]


if __name__ == "__main__":
    # Load and test with training examples
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/8597cfd7.json", "r") as f:
        data = json.load(f)
    
    print("Testing training examples:")
    all_pass = True
    for i, example in enumerate(data["train"]):
        input_grid = example["input"]
        expected = example["output"]
        result = solve(input_grid)
        
        passed = result == expected
        all_pass = all_pass and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"Example {i+1}: {status}")
        if not passed:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
    
    if all_pass:
        print("\n✓ All training examples passed!")
    else:
        print("\n✗ Some examples failed!")
    
    # Also test on test set to verify
    print("\nTest set predictions:")
    for i, example in enumerate(data["test"]):
        input_grid = example["input"]
        result = solve(input_grid)
        expected = example.get("output")
        if expected:
            match = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {match} {result}")
        else:
            print(f"Test {i+1}: {result}")
