def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Transformation rule:
    - 4 L-shaped corners define a rectangle (3 corners are color 8, 1 is color C)
    - Complete the rectangle border with 8s, fill interior with C
    - Extend horizontal/vertical bands outward from the rectangle, filled with C
    - Scattered pixels in the bands draw lines from themselves to the grid edge
    """
    rows = len(grid)
    cols = len(grid[0])

    # Find 8-cells to determine rectangle bounds
    eight_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
    r1 = min(r for r, c in eight_cells)
    r2 = max(r for r, c in eight_cells)
    c1 = min(c for r, c in eight_cells)
    c2 = max(c for r, c in eight_cells)

    # 4 corner L-shapes; one corner uses fill color C instead of 8
    corner_positions = {
        'TL': [(r1, c1), (r1, c1 + 1), (r1 + 1, c1)],
        'TR': [(r1, c2 - 1), (r1, c2), (r1 + 1, c2)],
        'BL': [(r2 - 1, c1), (r2, c1), (r2, c1 + 1)],
        'BR': [(r2 - 1, c2), (r2, c2 - 1), (r2, c2)],
    }

    corner_cells: set[tuple[int, int]] = set()
    fill_color = None
    for cells in corner_positions.values():
        for r, c in cells:
            corner_cells.add((r, c))
        for r, c in cells:
            v = grid[r][c]
            if v != 8 and v != 0:
                fill_color = v
                break

    result = [row[:] for row in grid]

    # Rectangle border = 8
    for c in range(c1, c2 + 1):
        result[r1][c] = 8
        result[r2][c] = 8
    for r in range(r1, r2 + 1):
        result[r][c1] = 8
        result[r][c2] = 8

    # Interior = fill color
    for r in range(r1 + 1, r2):
        for c in range(c1 + 1, c2):
            result[r][c] = fill_color

    # Horizontal bands = fill color
    for r in range(r1, r2 + 1):
        for c in range(0, c1):
            result[r][c] = fill_color
        for c in range(c2 + 1, cols):
            result[r][c] = fill_color

    # Vertical bands = fill color
    for r in range(0, r1):
        for c in range(c1, c2 + 1):
            result[r][c] = fill_color
    for r in range(r2 + 1, rows):
        for c in range(c1, c2 + 1):
            result[r][c] = fill_color

    # Scatter lines: each pixel draws a line from itself to the grid edge (away from rect)
    # Process from rect outward so further pixels overwrite tails of closer ones

    # Left horizontal band
    for r in range(r1, r2 + 1):
        pixels = sorted(
            [(c, grid[r][c]) for c in range(0, c1)
             if (r, c) not in corner_cells and grid[r][c] != 0],
            key=lambda x: -x[0],
        )
        for c, val in pixels:
            for cc in range(0, c + 1):
                result[r][cc] = val

    # Right horizontal band
    for r in range(r1, r2 + 1):
        pixels = sorted(
            [(c, grid[r][c]) for c in range(c2 + 1, cols)
             if (r, c) not in corner_cells and grid[r][c] != 0],
            key=lambda x: x[0],
        )
        for c, val in pixels:
            for cc in range(c, cols):
                result[r][cc] = val

    # Top vertical band
    for c in range(c1, c2 + 1):
        pixels = sorted(
            [(r, grid[r][c]) for r in range(0, r1)
             if (r, c) not in corner_cells and grid[r][c] != 0],
            key=lambda x: -x[0],
        )
        for r, val in pixels:
            for rr in range(0, r + 1):
                result[rr][c] = val

    # Bottom vertical band
    for c in range(c1, c2 + 1):
        pixels = sorted(
            [(r, grid[r][c]) for r in range(r2 + 1, rows)
             if (r, c) not in corner_cells and grid[r][c] != 0],
            key=lambda x: x[0],
        )
        for r, val in pixels:
            for rr in range(r, rows):
                result[rr][c] = val

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/256b0a75.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
