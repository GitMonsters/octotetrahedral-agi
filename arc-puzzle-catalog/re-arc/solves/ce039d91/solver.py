"""ARC-AGI puzzle ce039d91 solver.

Rule: A gray(5) cell becomes blue(1) if its horizontal mirror
(same row, column reflected across the vertical center line)
is also gray(5). Otherwise it stays gray(5).
"""


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])
    output = [row[:] for row in input_grid]
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] == 5:
                mirror_c = cols - 1 - c
                if input_grid[r][mirror_c] == 5:
                    output[r][c] = 1
    return output
