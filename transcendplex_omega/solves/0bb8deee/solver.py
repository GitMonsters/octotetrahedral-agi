def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find horizontal divider: a row where all cells are the same non-zero color
    h_div = -1
    for r in range(rows):
        if grid[r][0] != 0 and all(grid[r][c] == grid[r][0] for c in range(cols)):
            h_div = r
            break

    # Find vertical divider: a column where all cells are the same non-zero color
    v_div = -1
    for c in range(cols):
        if grid[0][c] != 0 and all(grid[r][c] == grid[0][c] for r in range(rows)):
            v_div = c
            break

    # Define 4 quadrants (excluding divider rows/cols)
    quadrants = [
        (range(0, h_div), range(0, v_div)),         # top-left
        (range(0, h_div), range(v_div + 1, cols)),   # top-right
        (range(h_div + 1, rows), range(0, v_div)),   # bottom-left
        (range(h_div + 1, rows), range(v_div + 1, cols)),  # bottom-right
    ]

    patches: list[list[list[int]]] = []
    for row_range, col_range in quadrants:
        # Find bounding box of non-zero cells in this quadrant
        min_r, max_r, min_c, max_c = rows, 0, cols, 0
        for r in row_range:
            for c in col_range:
                if grid[r][c] != 0:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
        # Extract 3x3 patch
        patch = []
        for r in range(min_r, max_r + 1):
            row = []
            for c in range(min_c, max_c + 1):
                row.append(grid[r][c])
            patch.append(row)
        patches.append(patch)

    # Assemble 6x6: top-left | top-right over bottom-left | bottom-right
    result = []
    for r in range(3):
        result.append(patches[0][r] + patches[1][r])
    for r in range(3):
        result.append(patches[2][r] + patches[3][r])
    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/0bb8deee.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
