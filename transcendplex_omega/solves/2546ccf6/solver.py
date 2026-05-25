def solve(grid: list[list[int]]) -> list[list[int]]:
    """Solve ARC task 2546ccf6.

    The grid is divided into cells by colored grid lines. For each non-grid
    color, three cells form an L-shape within a 2x2 block. The missing fourth
    cell is filled with the 180° rotation of the diagonally opposite cell.
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find grid color from a full-span horizontal line
    grid_color = None
    for r in range(rows):
        if grid[r][0] != 0 and all(v == grid[r][0] for v in grid[r]):
            grid_color = grid[r][0]
            break

    h_lines = [r for r in range(rows) if all(grid[r][c] == grid_color for c in range(cols))]
    v_lines = [c for c in range(cols) if all(grid[r][c] == grid_color for r in range(rows))]

    def build_groups(lines: list[int], total: int) -> list[tuple[int, int]]:
        groups: list[tuple[int, int]] = []
        prev = 0
        for line in lines:
            if line > prev:
                groups.append((prev, line - 1))
            prev = line + 1
        if prev < total:
            groups.append((prev, total - 1))
        return groups

    row_groups = build_groups(h_lines, rows)
    col_groups = build_groups(v_lines, cols)
    nr, nc = len(row_groups), len(col_groups)

    def get_cell(ri: int, ci: int) -> list[list[int]]:
        r0, r1 = row_groups[ri]
        c0, c1 = col_groups[ci]
        return [[grid[r][c] for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]

    def set_cell(ri: int, ci: int, cell: list[list[int]]) -> None:
        r0, r1 = row_groups[ri]
        c0, c1 = col_groups[ci]
        for dr, r in enumerate(range(r0, r1 + 1)):
            for dc, c in enumerate(range(c0, c1 + 1)):
                result[r][c] = cell[dr][dc]

    # Map each non-grid color to the set of cell positions containing it
    color_cells: dict[int, set[tuple[int, int]]] = {}
    for ri in range(nr):
        for ci in range(nc):
            cell = get_cell(ri, ci)
            for row in cell:
                for v in row:
                    if v != 0 and v != grid_color:
                        color_cells.setdefault(v, set()).add((ri, ci))

    # For each color: find 2x2 L-shape, fill missing cell with rot180 of diagonal
    for color, cell_set in color_cells.items():
        if len(cell_set) != 3:
            continue
        cells = list(cell_set)
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        if max_r - min_r != 1 or max_c - min_c != 1:
            continue
        all_four = {(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)}
        missing = (all_four - cell_set).pop()
        diag_r = max_r if missing[0] == min_r else min_r
        diag_c = max_c if missing[1] == min_c else min_c
        filled = [row[::-1] for row in get_cell(diag_r, diag_c)[::-1]]
        set_cell(missing[0], missing[1], filled)

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/2546ccf6.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
