def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find separator rows and cols (rows/cols containing 2s)
    sep_rows = sorted({r for r in range(rows) for c in range(cols) if grid[r][c] == 2})
    sep_cols = sorted({c for r in range(rows) for c in range(cols) if grid[r][c] == 2})

    # Add virtual boundaries at grid edges
    row_bounds = [-1] + sep_rows + [rows]
    col_bounds = [-1] + sep_cols + [cols]

    # Build output: start with zeros, place 2s back
    result = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                result[r][c] = 2

    # Process each cell between consecutive boundaries
    for ri in range(len(row_bounds) - 1):
        for ci in range(len(col_bounds) - 1):
            r_start = row_bounds[ri] + 1
            r_end = row_bounds[ri + 1]
            c_start = col_bounds[ci] + 1
            c_end = col_bounds[ci + 1]
            H = r_end - r_start
            W = c_end - c_start
            if H <= 0 or W <= 0:
                continue

            # Extract 1-pixels within this cell
            pixels = []
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    if grid[r][c] == 1:
                        pixels.append((r - r_start, c - c_start))

            if not pixels:
                continue

            # Bounding box of the shape
            min_r = min(p[0] for p in pixels)
            max_r = max(p[0] for p in pixels)
            min_c = min(p[1] for p in pixels)
            max_c = max(p[1] for p in pixels)
            h = max_r - min_r + 1
            w = max_c - min_c + 1

            # Center the shape within the cell
            new_top = (H - h) // 2
            new_left = (W - w) // 2

            for (pr, pc) in pixels:
                nr = new_top + (pr - min_r) + r_start
                nc = new_left + (pc - min_c) + c_start
                result[nr][nc] = 1

    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/20981f0e.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
