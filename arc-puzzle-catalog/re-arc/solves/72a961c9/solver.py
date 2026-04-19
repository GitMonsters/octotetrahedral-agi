import copy


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    """Each non-1 marker in the horizontal line shoots a perpendicular arm upward.
    Color 2 (red) -> arm length 4, Color 8 (cyan) -> arm length 3.
    Arm tip = marker color, stem = blue (1)."""
    output = copy.deepcopy(input_grid)
    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find the line row (first row with non-zero values)
    line_row = None
    for r in range(rows):
        if any(input_grid[r][c] != 0 for c in range(cols)):
            line_row = r
            break

    if line_row is None:
        return output

    for c in range(cols):
        v = input_grid[line_row][c]
        if v not in (0, 1):
            arm_length = 4 if v == 2 else 3
            arm_length = min(arm_length, line_row)  # cap by available space

            for i in range(1, arm_length + 1):
                if i == arm_length:
                    output[line_row - i][c] = v  # marker color at tip
                else:
                    output[line_row - i][c] = 1  # blue fill

    return output
