"""ARC puzzle e345f17b solver.
Rule: Split input horizontally into left and right halves.
Where BOTH halves are 0 (background), output 4 (yellow). Otherwise 0.
"""

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0]) // 2
    output = []
    for r in range(rows):
        row = []
        for c in range(cols):
            left = input_grid[r][c]
            right = input_grid[r][c + cols]
            row.append(4 if left == 0 and right == 0 else 0)
        output.append(row)
    return output


if __name__ == "__main__":
    examples = [
        ([[6,6,6,6,5,0,5,0],[6,0,0,0,5,5,0,0],[6,0,6,6,0,0,5,5],[0,0,6,0,0,5,5,0]],
         [[0,0,0,0],[0,0,4,4],[0,4,0,0],[4,0,0,4]]),
        ([[0,6,6,0,5,5,5,0],[0,6,0,6,5,0,0,5],[0,6,6,6,5,5,5,5],[6,0,0,0,0,5,0,5]],
         [[0,0,0,4],[0,0,4,0],[0,0,0,0],[0,0,4,0]]),
        ([[6,6,6,0,5,0,5,5],[6,0,0,0,0,5,5,5],[6,0,0,0,0,0,0,0],[0,6,6,6,5,5,0,0]],
         [[0,0,0,0],[0,0,0,0],[0,4,4,4],[0,0,0,0]]),
        ([[6,0,6,0,0,0,5,5],[0,6,6,6,5,0,5,5],[6,6,0,6,5,0,5,5],[6,6,0,0,5,0,0,0]],
         [[0,4,0,0],[0,0,0,0],[0,0,0,0],[0,0,4,4]]),
    ]

    all_pass = True
    for i, (inp, expected) in enumerate(examples):
        result = transform(inp)
        ok = result == expected
        print(f"Example {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
            all_pass = False

    test_input = [[6,0,6,6,5,0,0,5],[0,0,0,6,5,5,5,5],[0,6,6,0,5,5,0,5],[6,6,0,0,5,5,5,0]]
    test_expected = [[0,4,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,4]]
    test_result = transform(test_input)
    test_ok = test_result == test_expected
    print(f"Test:      {'PASS' if test_ok else 'FAIL'}")
    if not test_ok:
        print(f"  Expected: {test_expected}")
        print(f"  Got:      {test_result}")
        all_pass = all_pass and test_ok

    print("SOLVED" if all_pass else "FAILED")
