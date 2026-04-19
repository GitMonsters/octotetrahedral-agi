"""ARC puzzle 5623160b solver.

Rule: Purple(9) cells act as anchors. Each non-purple, non-background object
adjacent to a purple cell gets pushed away from that purple cell, sliding to
the grid boundary while preserving its shape. Purple stays in place.
"""

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])
    bg = 7
    pivot = 9

    output = [[bg] * cols for _ in range(rows)]

    # Place purple cells (they stay fixed)
    purple_set = set()
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] == pivot:
                purple_set.add((r, c))
                output[r][c] = pivot

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] == bg or input_grid[r][c] == pivot:
                visited[r][c] = True

    def flood_fill(sr, sc):
        color = input_grid[sr][sc]
        stack = [(sr, sc)]
        visited[sr][sc] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and input_grid[nr][nc] == color:
                    visited[nr][nc] = True
                    stack.append((nr, nc))
        return cells, color

    objects = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c]:
                cells, color = flood_fill(r, c)
                objects.append((cells, color))

    for cells, color in objects:
        direction = None
        for r, c in cells:
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if (nr, nc) in purple_set:
                    # Direction from purple to object cell is (-dr, -dc)
                    mdr, mdc = -dr, -dc
                    if mdr == -1:
                        direction = 'UP'
                    elif mdr == 1:
                        direction = 'DOWN'
                    elif mdc == -1:
                        direction = 'LEFT'
                    else:
                        direction = 'RIGHT'
                    break
            if direction:
                break

        if direction is None:
            for r, c in cells:
                output[r][c] = color
            continue

        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        if direction == 'UP':
            sr, sc = -min_r, 0
        elif direction == 'DOWN':
            sr, sc = (rows - 1 - max_r), 0
        elif direction == 'LEFT':
            sr, sc = 0, -min_c
        else:
            sr, sc = 0, (cols - 1 - max_c)

        for r, c in cells:
            output[r + sr][c + sc] = color

    return output


# --- Test against all training examples ---
if __name__ == "__main__":
    examples = [
        (
            [[7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,1,7,7,7,7],[7,7,7,7,7,1,1,9,0,0],[7,7,7,7,7,7,7,9,7,7],[7,7,7,7,7,7,6,6,6,7],[7,7,7,7,7,7,7,6,7,7],[7,2,9,7,7,7,7,7,7,7],[7,7,8,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7]],
            [[7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7],[1,7,7,7,7,7,7,7,7,7],[1,1,7,7,7,7,7,9,0,0],[7,7,7,7,7,7,7,9,7,7],[7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7],[2,7,9,7,7,7,7,7,7,7],[7,7,7,7,7,7,6,6,6,7],[7,7,8,7,7,7,7,6,7,7]]
        ),
        (
            [[7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7],[7,7,7,5,7,7,7,7,7],[7,7,7,9,2,2,7,7,7],[7,7,7,8,7,2,7,7,7],[7,7,7,8,7,7,7,7,7],[7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7]],
            [[7,7,7,5,7,7,7,7,7],[7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7],[7,7,7,9,7,7,7,2,2],[7,7,7,7,7,7,7,7,2],[7,7,7,7,7,7,7,7,7],[7,7,7,8,7,7,7,7,7],[7,7,7,8,7,7,7,7,7]]
        ),
        (
            [[7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,4,7,4,7,7,7,7,7],[7,7,7,7,4,4,4,7,7,7,7,7],[7,7,7,7,7,9,7,7,7,7,7,7],[7,7,7,6,7,9,7,7,7,7,7,7],[7,7,7,9,9,9,3,3,7,7,7,7],[7,7,7,8,7,9,3,3,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7]],
            [[7,7,7,6,4,7,4,7,7,7,7,7],[7,7,7,7,4,4,4,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,9,7,7,7,7,7,7],[7,7,7,7,7,9,7,7,7,7,7,7],[7,7,7,9,9,9,7,7,7,7,3,3],[7,7,7,7,7,9,7,7,7,7,3,3],[7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,8,7,7,7,7,7,7,7,7]]
        ),
    ]

    test_input = [[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,9,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,9,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,9,9,7,7],[7,7,5,5,5,5,7,5,5,7,7,7,7,7,7,7],[7,7,5,7,7,5,5,5,7,7,7,7,7,7,7,7],[7,7,5,7,7,7,7,5,5,5,7,7,7,7,7,7],[7,7,5,5,5,7,7,5,7,5,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,9,9,8,7,7,7,7],[7,7,7,7,7,7,7,7,3,9,9,8,7,7,7,7],[7,7,7,9,7,7,7,7,7,6,0,7,7,7,7,7],[7,7,7,9,7,7,7,7,7,7,0,0,7,7,7,7],[7,9,9,9,9,9,7,7,7,7,7,0,7,0,7,7],[7,4,7,9,7,7,7,7,7,7,0,0,0,0,7,7],[7,7,2,9,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]]
    expected = [[7,7,5,5,5,5,7,5,5,7,7,7,7,7,7,7],[7,7,5,7,7,5,5,5,7,7,7,7,9,7,7,7],[7,7,5,7,7,7,7,5,5,5,7,7,9,7,7,7],[7,7,5,5,5,7,7,5,7,5,7,7,9,9,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,9,9,7,7,7,7,8],[3,7,7,7,7,7,7,7,7,9,9,7,7,7,7,8],[7,7,7,9,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,9,7,7,7,7,7,7,7,7,7,7,7,7],[7,9,9,9,9,9,7,7,7,7,0,7,7,7,7,7],[7,7,7,9,7,7,7,7,7,7,0,0,7,7,7,7],[2,7,7,9,7,7,7,7,7,7,7,0,7,0,7,7],[7,4,7,7,7,7,7,7,7,6,0,0,0,0,7,7]]

    all_pass = True
    for i, (inp, exp) in enumerate(examples):
        result = transform(inp)
        if result == exp:
            print(f"Example {i}: PASS")
        else:
            print(f"Example {i}: FAIL")
            for r in range(len(exp)):
                if result[r] != exp[r]:
                    print(f"  Row {r}: got  {result[r]}")
                    print(f"         want {exp[r]}")
            all_pass = False

    test_result = transform(test_input)
    if test_result == expected:
        print("Test: PASS")
    else:
        print("Test: FAIL")
        for r in range(len(expected)):
            if test_result[r] != expected[r]:
                print(f"  Row {r}: got  {test_result[r]}")
                print(f"         want {expected[r]}")
        all_pass = False

    print("SOLVED" if all_pass else "FAILED")
