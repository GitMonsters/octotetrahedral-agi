def solve(grid: list[list[int]]) -> list[list[int]]:
    N = len(grid)
    # Period P = max color (colors are 1..P)
    P = max(v for row in grid for v in row)

    # Find offset_B from a non-zero cell we can classify
    # Row 0: all cells have c >= r=0, so they're in the "right tiling"
    offset_B = None
    for r in range(N):
        if offset_B is not None:
            break
        for c in range(N):
            if grid[r][c] != 0:
                if c >= r:
                    offset_B = (grid[r][c] - 1 - c) % P
                else:
                    offset_B = (grid[r][c] - 2 - c) % P
                break

    # Build output: diagonal staircase at c = r
    # Right tiling (c >= r): value = ((c + offset_B) % P) + 1
    # Left  tiling (c <  r): value = ((c + offset_B + 1) % P) + 1
    result = []
    for r in range(N):
        row = []
        for c in range(N):
            if c < r:
                v = ((c + offset_B + 1) % P) + 1
            else:
                v = ((c + offset_B) % P) + 1
            row.append(v)
        result.append(row)
    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/1e97544e.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
