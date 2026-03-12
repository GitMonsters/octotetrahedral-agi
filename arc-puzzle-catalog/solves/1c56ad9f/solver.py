def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find vertical bounding box of non-zero cells
    min_r, max_r = rows, -1
    for r in range(rows):
        if any(grid[r][c] != 0 for c in range(cols)):
            if r < min_r: min_r = r
            max_r = r

    n = max_r - min_r + 1  # shape height
    # Zigzag cycle: 0, -1, 0, +1  with start phase chosen so last row offset = 0
    start = (3 - n) % 4
    cycle = [0, -1, 0, 1]

    result = [[0] * cols for _ in range(rows)]
    for r in range(min_r, max_r + 1):
        offset = cycle[(r - min_r + start) % 4]
        for c in range(cols):
            if grid[r][c] != 0:
                nc = c + offset
                if 0 <= nc < cols:
                    result[r][nc] = grid[r][c]
    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/1c56ad9f.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
