def solve(grid: list[list[int]]) -> list[list[int]]:
    import copy
    result = copy.deepcopy(grid)
    rows = len(grid)
    cols = len(grid[0])
    last_row = rows - 1

    # Find columns marked with 5 in the last row
    marked_cols = [c for c in range(cols) if grid[last_row][c] == 5]

    for c in marked_cols:
        # Collect all non-zero values in this column, sorted by row
        values = [(r, grid[r][c]) for r in range(rows) if grid[r][c] != 0]

        # Fill segments: each value fills upward from previous value's row+1 to its own row
        prev_row = -1
        for r, v in values:
            for fill_r in range(prev_row + 1, r + 1):
                result[fill_r][c] = v
            prev_row = r

    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/17b80ad2.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
        if status == "FAIL":
            for r in range(len(result)):
                if result[r] != ex["output"][r]:
                    print(f"  Row {r}: got {result[r]}")
                    print(f"  Row {r}: exp {ex['output'][r]}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
