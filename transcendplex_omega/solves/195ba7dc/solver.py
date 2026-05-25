def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    # Column 6 is separator (value 2). Left=cols 0-5, Right=cols 7-12.
    result = []
    for r in range(rows):
        row = []
        for c in range(6):
            left = 1 if grid[r][c] == 7 else 0
            right = 1 if grid[r][c + 7] == 7 else 0
            row.append(1 if (left or right) else 0)
        result.append(row)
    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/195ba7dc.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
