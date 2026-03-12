def solve(grid: list[list[int]]) -> list[list[int]]:
    # Find the separator row (all 7s)
    sep = next(i for i, row in enumerate(grid) if all(c == 7 for c in row))
    top = grid[:sep]
    bottom = grid[sep + 1:]
    # NOR: output 8 where both halves are 0, else 0
    return [
        [8 if top[r][c] == 0 and bottom[r][c] == 0 else 0
         for c in range(len(top[0]))]
        for r in range(len(top))
    ]

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/0c9aba6e.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
