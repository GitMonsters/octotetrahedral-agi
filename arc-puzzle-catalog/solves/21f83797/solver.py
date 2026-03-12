def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    # Find the two 2-pixels
    pts = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]
    (r1, c1), (r2, c2) = pts[0], pts[1]
    rmin, rmax = min(r1, r2), max(r1, r2)
    cmin, cmax = min(c1, c2), max(c1, c2)

    result = [[0]*cols for _ in range(rows)]
    # Draw full cross lines with 2
    for r in range(rows):
        result[r][c1] = 2
        result[r][c2] = 2
    for c in range(cols):
        result[r1][c] = 2
        result[r2][c] = 2
    # Fill interior rectangle with 1
    for r in range(rmin+1, rmax):
        for c in range(cmin+1, cmax):
            result[r][c] = 1
    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/21f83797.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
