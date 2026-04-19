"""ARC-AGI puzzle bae5c565 solver.

Rule: Row 0 contains a color pattern with background(5) at the cyan column position.
A vertical cyan(8) line acts as the axis. The pattern from row 0 is "draped" as an
expanding triangle from the top of the cyan line downward, adding one more column
on each side per row, capped at the grid edges. Row 0 is cleared to background.
"""

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])
    bg = 5

    # Find the vertical cyan(8) line (skip row 0 which is the pattern)
    cyan_col = None
    cyan_start = None
    for c in range(cols):
        start = None
        count = 0
        for r in range(1, rows):
            if input_grid[r][c] == 8:
                if start is None:
                    start = r
                count += 1
        if count >= 2:
            cyan_col = c
            cyan_start = start
            break

    pattern = input_grid[0][:]

    # Build output filled with background
    output = [[bg] * cols for _ in range(rows)]

    # Place cyan line and expanding pattern
    for r in range(cyan_start, rows):
        output[r][cyan_col] = 8
        expansion = r - cyan_start
        for i in range(1, min(expansion, cyan_col) + 1):
            output[r][cyan_col - i] = pattern[cyan_col - i]
        for i in range(1, min(expansion, cols - 1 - cyan_col) + 1):
            output[r][cyan_col + i] = pattern[cyan_col + i]

    return output


if __name__ == "__main__":
    examples = [
        (
            [[2,2,7,1,9,1,5,8,6,0,3,2,2],[5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,8,5,5,5,5,5,5],[5,5,5,5,5,5,8,5,5,5,5,5,5],[5,5,5,5,5,5,8,5,5,5,5,5,5],[5,5,5,5,5,5,8,5,5,5,5,5,5],[5,5,5,5,5,5,8,5,5,5,5,5,5],[5,5,5,5,5,5,8,5,5,5,5,5,5],[5,5,5,5,5,5,8,5,5,5,5,5,5],[5,5,5,5,5,5,8,5,5,5,5,5,5]],
            [[5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,8,5,5,5,5,5,5],[5,5,5,5,5,1,8,8,5,5,5,5,5],[5,5,5,5,9,1,8,8,6,5,5,5,5],[5,5,5,1,9,1,8,8,6,0,5,5,5],[5,5,7,1,9,1,8,8,6,0,3,5,5],[5,2,7,1,9,1,8,8,6,0,3,2,5],[2,2,7,1,9,1,8,8,6,0,3,2,2],[2,2,7,1,9,1,8,8,6,0,3,2,2]]
        ),
        (
            [[0,1,6,9,5,9,6,1,0],[5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5],[5,5,5,5,8,5,5,5,5],[5,5,5,5,8,5,5,5,5],[5,5,5,5,8,5,5,5,5],[5,5,5,5,8,5,5,5,5],[5,5,5,5,8,5,5,5,5],[5,5,5,5,8,5,5,5,5]],
            [[5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5],[5,5,5,5,8,5,5,5,5],[5,5,5,9,8,9,5,5,5],[5,5,6,9,8,9,6,5,5],[5,1,6,9,8,9,6,1,5],[0,1,6,9,8,9,6,1,0],[0,1,6,9,8,9,6,1,0]]
        ),
    ]

    test_input = [[4,6,7,2,9,5,3,3,4,3,3],[5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,8,5,5,5,5,5],[5,5,5,5,5,8,5,5,5,5,5],[5,5,5,5,5,8,5,5,5,5,5],[5,5,5,5,5,8,5,5,5,5,5],[5,5,5,5,5,8,5,5,5,5,5],[5,5,5,5,5,8,5,5,5,5,5]]
    test_expected = [[5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,8,5,5,5,5,5],[5,5,5,5,9,8,3,5,5,5,5],[5,5,5,2,9,8,3,3,5,5,5],[5,5,7,2,9,8,3,3,4,5,5],[5,6,7,2,9,8,3,3,4,3,5],[4,6,7,2,9,8,3,3,4,3,3]]

    all_pass = True
    for i, (inp, exp) in enumerate(examples):
        result = transform(inp)
        if result == exp:
            print(f"Example {i}: PASS")
        else:
            print(f"Example {i}: FAIL")
            for r in range(len(exp)):
                if result[r] != exp[r]:
                    print(f"  Row {r}: got {result[r]}")
                    print(f"         exp {exp[r]}")
            all_pass = False

    test_result = transform(test_input)
    if test_result == test_expected:
        print("Test: PASS")
    else:
        print("Test: FAIL")
        for r in range(len(test_expected)):
            if test_result[r] != test_expected[r]:
                print(f"  Row {r}: got {test_result[r]}")
                print(f"         exp {test_expected[r]}")
        all_pass = False

    print("SOLVED" if all_pass else "FAILED")
