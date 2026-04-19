from collections import Counter

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])

    # Compute column modes (for vertical stripes hypothesis)
    col_modes = []
    for c in range(cols):
        col_vals = [input_grid[r][c] for r in range(rows)]
        col_modes.append(Counter(col_vals).most_common(1)[0][0])

    # Compute row modes (for horizontal stripes hypothesis)
    row_modes = []
    for r in range(rows):
        row_modes.append(Counter(input_grid[r]).most_common(1)[0][0])

    # Count how many input cells match each hypothesis
    v_match = sum(1 for r in range(rows) for c in range(cols) if col_modes[c] == input_grid[r][c])
    h_match = sum(1 for r in range(rows) for c in range(cols) if row_modes[r] == input_grid[r][c])

    if v_match >= h_match:
        return [col_modes[:] for _ in range(rows)]
    else:
        return [[row_modes[r]] * cols for r in range(rows)]
