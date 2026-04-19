"""ARC puzzle 25d487eb solver.

Rule: A triangle/diamond shape has a single marker pixel of a different color.
The marker sits on the edge of the shape. From the marker, a line is drawn
in the direction *into* the shape (opposite the open/empty side), filling
all background (0) cells with the marker's color to the grid edge.
"""

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    import copy
    rows = len(input_grid)
    cols = len(input_grid[0])
    grid = copy.deepcopy(input_grid)

    # Find all non-zero colors and their positions
    color_counts: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0:
                color_counts.setdefault(v, []).append((r, c))

    # Marker is the color that appears exactly once
    marker_color = None
    marker_pos = None
    for color, positions in color_counts.items():
        if len(positions) == 1:
            marker_color = color
            marker_pos = positions[0]
            break

    if marker_pos is None:
        return grid

    mr, mc = marker_pos

    # Find the open direction (neighbor that is 0)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    open_dir = None
    for dr, dc in directions:
        nr, nc = mr + dr, mc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] == 0:
                open_dir = (dr, dc)
                break
        else:
            # Out of bounds counts as open
            open_dir = (dr, dc)
            break

    if open_dir is None:
        return grid

    # Shoot line in OPPOSITE direction
    shoot_dr, shoot_dc = -open_dir[0], -open_dir[1]
    r, c = mr + shoot_dr, mc + shoot_dc
    while 0 <= r < rows and 0 <= c < cols:
        if grid[r][c] == 0:
            grid[r][c] = marker_color
        r += shoot_dr
        c += shoot_dc

    return grid


# --- Test against all training examples ---
if __name__ == "__main__":
    examples = [
        (
            [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,2,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,2,2,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,2,2,0,0,0,0,0,0,0,0,0],[0,0,0,2,2,0,0,0,0,0,0,0,0,0,0],[0,0,0,2,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,2,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,2,2,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,2,2,1,1,1,1,1,1,1,1,1],[0,0,0,2,2,0,0,0,0,0,0,0,0,0,0],[0,0,0,2,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        ),
        (
            [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,8,0,0,0,0,0],[0,0,0,0,0,8,8,8,0,0,0,0],[0,0,0,0,8,8,8,8,8,0,0,0],[0,0,0,8,8,8,3,8,8,8,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,3,0,0,0,0,0],[0,0,0,0,0,0,3,0,0,0,0,0],[0,0,0,0,0,0,3,0,0,0,0,0],[0,0,0,0,0,0,3,0,0,0,0,0],[0,0,0,0,0,0,3,0,0,0,0,0],[0,0,0,0,0,0,8,0,0,0,0,0],[0,0,0,0,0,8,8,8,0,0,0,0],[0,0,0,0,8,8,8,8,8,0,0,0],[0,0,0,8,8,8,3,8,8,8,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]],
        ),
        (
            [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,3,3,2,3,3,0,0,0,0,0],[0,0,0,3,3,3,0,0,0,0,0,0],[0,0,0,0,3,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,3,3,2,3,3,0,0,0,0,0],[0,0,0,3,3,3,0,0,0,0,0,0],[0,0,0,0,3,0,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0,0,0]],
        ),
    ]

    test_input = [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,4,0,0,0,0,0,0],[0,0,0,4,4,4,0,0,0,0,0],[0,0,4,4,4,4,4,0,0,0,0],[0,4,4,4,8,4,4,4,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]
    test_expected = [[0,0,0,0,8,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0,0,0],[0,0,0,0,4,0,0,0,0,0,0],[0,0,0,4,4,4,0,0,0,0,0],[0,0,4,4,4,4,4,0,0,0,0],[0,4,4,4,8,4,4,4,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]

    all_pass = True
    for i, (inp, expected) in enumerate(examples):
        result = transform(inp)
        if result == expected:
            print(f"Example {i}: PASS")
        else:
            print(f"Example {i}: FAIL")
            all_pass = False

    test_result = transform(test_input)
    if test_result == test_expected:
        print("Test: PASS")
    else:
        print("Test: FAIL")
        all_pass = False

    print("SOLVED" if all_pass else "FAILED")
