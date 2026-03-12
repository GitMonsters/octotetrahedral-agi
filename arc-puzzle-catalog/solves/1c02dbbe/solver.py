def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find bounding box of 5 cells (the rectangle)
    r_min, r_max = rows, 0
    c_min, c_max = cols, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                r_min = min(r_min, r)
                r_max = max(r_max, r)
                c_min = min(c_min, c)
                c_max = max(c_max, c)

    # Find all colored markers grouped by color
    colors: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0 and v != 5:
                colors.setdefault(v, []).append((r, c))

    # Build output: rectangle filled with 5, rest 0
    result = [[0] * cols for _ in range(rows)]
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            result[r][c] = 5

    # Each color has 3 markers: 1 corner (inside rect), 2 edge (outside rect)
    for color, markers in colors.items():
        corner = None
        row_split = None
        col_split = None

        for r, c in markers:
            if r_min <= r <= r_max and c_min <= c <= c_max:
                corner = (r, c)
            elif r < r_min or r > r_max:
                col_split = c  # top/bottom marker → column boundary
            else:
                row_split = r  # left/right marker → row boundary

        cr, cc = corner

        if cr == r_min and cc == c_min:      # top-left
            rs, re = r_min, row_split
            cs, ce = c_min, col_split
        elif cr == r_min and cc == c_max:     # top-right
            rs, re = r_min, row_split
            cs, ce = col_split, c_max
        elif cr == r_max and cc == c_min:     # bottom-left
            rs, re = row_split, r_max
            cs, ce = c_min, col_split
        else:                                  # bottom-right
            rs, re = row_split, r_max
            cs, ce = col_split, c_max

        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                result[r][c] = color

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/1c02dbbe.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
