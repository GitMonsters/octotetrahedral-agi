"""ARC Puzzle 20270e3b solver.

Pattern: Two yellow+orange shapes on blue background. Orange marks where they
were "cut apart". Reconnect by shifting the smaller shape so its orange aligns
one step toward the larger shape's body from the larger shape's orange.
"""
from collections import deque


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    grid = input_grid
    rows = len(grid)
    cols = len(grid[0])
    bg = 1

    # Find connected components of non-background cells
    visited = [[False] * cols for _ in range(rows)]
    components: list[list[tuple[int, int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                comp: list[tuple[int, int, int]] = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc, grid[cr][cc]))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != bg:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                components.append(comp)

    # A = larger component, B = smaller
    components.sort(key=lambda c: len(c), reverse=True)
    comp_a, comp_b = components[0], components[1]

    a_orange = [(r, c) for r, c, v in comp_a if v == 7]
    a_yellow_set = {(r, c) for r, c, v in comp_a if v == 4}

    # Dock direction: from A's orange toward A's yellow body
    dock_dir = None
    for o_r, o_c in a_orange:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (o_r + dr, o_c + dc) in a_yellow_set:
                dock_dir = (dr, dc)
                break
        if dock_dir:
            break

    b_orange = [(r, c) for r, c, v in comp_b if v == 7]

    # Shift B so its orange lands at (A_orange + dock_dir)
    a_o = a_orange[0]
    b_o = b_orange[0]
    target_r = a_o[0] + dock_dir[0]
    target_c = a_o[1] + dock_dir[1]
    shift_r = target_r - b_o[0]
    shift_c = target_c - b_o[1]

    # Union of all cells → yellow(4)
    combined: set[tuple[int, int]] = set()
    for r, c, v in comp_a:
        combined.add((r, c))
    for r, c, v in comp_b:
        combined.add((r + shift_r, c + shift_c))

    # Bounding box
    min_r = min(r for r, c in combined)
    max_r = max(r for r, c in combined)
    min_c = min(c for r, c in combined)
    max_c = max(c for r, c in combined)

    out_rows = max_r - min_r + 1
    out_cols = max_c - min_c + 1
    output = [[bg] * out_cols for _ in range(out_rows)]
    for r, c in combined:
        output[r - min_r][c - min_c] = 4

    return output


# ── Testing ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    examples = [
        {
            "input": [[4,4,4,4,4,4,4,4,1,7,7,7,1],[4,1,1,7,7,7,1,4,1,4,4,4,4],[4,1,1,1,1,1,1,4,1,4,1,1,4],[4,1,1,1,1,1,1,4,1,4,1,1,4],[4,1,1,1,1,1,1,4,1,1,4,4,1],[4,1,1,1,1,1,1,4,1,1,1,1,1],[4,4,4,4,4,4,4,4,1,1,1,1,1]],
            "output": [[4,4,4,4,4,4,4,4],[4,1,1,4,4,4,4,4],[4,1,1,4,1,1,4,4],[4,1,1,4,1,1,4,4],[4,1,1,1,4,4,1,4],[4,1,1,1,1,1,1,4],[4,4,4,4,4,4,4,4]],
        },
        {
            "input": [[4,1,1,1,1,1,1,1,1,1,7,1,4],[4,4,4,4,4,4,1,1,1,1,4,4,4],[1,1,1,4,1,4,1,1,1,1,1,1,1],[1,4,4,4,4,4,1,1,1,1,1,1,1],[1,4,1,1,1,1,1,1,1,1,1,1,1],[1,4,1,1,1,1,1,1,1,1,1,1,1],[1,7,1,1,1,1,1,1,1,1,1,1,1]],
            "output": [[4,1,1,1,1,1],[4,4,4,4,4,4],[1,1,1,4,1,4],[1,4,4,4,4,4],[1,4,1,1,1,1],[1,4,1,4,1,1],[1,4,4,4,1,1]],
        },
        {
            "input": [[4,4,4],[4,1,4],[4,4,4],[7,7,7],[1,1,1],[7,7,7],[4,4,4],[4,1,4],[4,4,4]],
            "output": [[4,4,4],[4,1,4],[4,4,4],[4,4,4],[4,1,4],[4,4,4]],
        },
        {
            "input": [[4,4,4,4,1,1,1,1,1],[4,1,1,4,1,1,1,1,1],[4,4,4,4,4,4,1,1,1],[1,1,1,1,1,4,1,1,1],[1,1,1,1,1,7,1,4,4],[1,1,7,1,1,1,1,4,1],[1,1,4,4,4,4,4,4,1]],
            "output": [[4,4,4,4,1,1,1,1,1,1,1,1],[4,1,1,4,1,1,1,1,1,1,1,1],[4,4,4,4,4,4,1,1,1,1,4,4],[1,1,1,1,1,4,1,1,1,1,4,1],[1,1,1,1,1,4,4,4,4,4,4,1]],
        },
    ]

    test_input = [[4,4,4,4,4,4,4,4,4,4],[4,1,4,1,4,1,4,7,4,1],[4,1,4,1,4,1,4,1,4,1],[4,1,4,1,4,1,4,1,4,1],[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1],[1,1,1,7,1,1,1,1,1,1],[1,4,1,4,1,1,1,1,1,1],[1,4,1,4,1,1,1,1,1,1],[1,4,1,4,1,1,1,1,1,1],[1,4,4,4,1,1,1,1,1,1]]
    expected_output = [[4,4,4,4,4,4,4,4,4,4],[4,1,4,1,4,4,4,4,4,1],[4,1,4,1,4,4,4,4,4,1],[4,1,4,1,4,4,4,4,4,1],[1,1,1,1,1,4,4,4,1,1]]

    all_pass = True
    for i, ex in enumerate(examples):
        result = transform(ex["input"])
        ok = result == ex["output"]
        print(f"Example {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            print(f"  Expected: {ex['output']}")
            print(f"  Got:      {result}")

    test_result = transform(test_input)
    test_ok = test_result == expected_output
    print(f"Test:      {'PASS' if test_ok else 'FAIL'}")
    if not test_ok:
        all_pass = False
        print(f"  Expected: {expected_output}")
        print(f"  Got:      {test_result}")

    print("\nSOLVED" if all_pass else "\nFAILED")
