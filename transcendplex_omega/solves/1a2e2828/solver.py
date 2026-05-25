def solve(grid: list[list[int]]) -> list[list[int]]:
    """Find the non-zero color that forms a complete uninterrupted line
    (an entire row or column where every cell is that same color).
    The grid has colored "bars" forming a cross-hatch pattern with a
    z-order hierarchy; the topmost bar is never interrupted by others."""
    rows = len(grid)
    cols = len(grid[0])

    for r in range(rows):
        val = grid[r][0]
        if val != 0 and all(grid[r][c] == val for c in range(cols)):
            return [[val]]

    for c in range(cols):
        val = grid[0][c]
        if val != 0 and all(grid[r][c] == val for r in range(rows)):
            return [[val]]

    return [[0]]


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/1a2e2828.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
