def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find bounding box of the 8-filled rectangle
    min_r, max_r, min_c, max_c = rows, 0, cols, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    center_r = (min_r + max_r) / 2.0
    center_c = (min_c + max_c) / 2.0

    # Find all 2x2 colored blocks (non-0, non-8)
    found: dict[int, tuple[float, float]] = {}
    for r in range(rows - 1):
        for c in range(cols - 1):
            v = grid[r][c]
            if v not in (0, 8):
                if (grid[r][c+1] == v and grid[r+1][c] == v and grid[r+1][c+1] == v):
                    found[v] = (r + 0.5, c + 0.5)

    # Place each color in the appropriate quadrant
    result = [[0, 0], [0, 0]]
    for color, (cr, cc) in found.items():
        qr = 0 if cr < center_r else 1
        qc = 0 if cc < center_c else 1
        result[qr][qc] = color

    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/19bb5feb.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
        if status == "FAIL":
            print(f"  Expected: {ex['output']}")
            print(f"  Got:      {result}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])} -> {result}")
