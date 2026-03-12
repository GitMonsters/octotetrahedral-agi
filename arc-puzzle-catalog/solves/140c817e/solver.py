def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    # Find background color and 1-positions
    ones: list[tuple[int, int]] = []
    bg = -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                ones.append((r, c))
            else:
                bg = grid[r][c]

    result = [[bg] * cols for _ in range(rows)]

    # Draw cross lines (value 1) through each original 1 position
    for r, c in ones:
        for cc in range(cols):
            result[r][cc] = 1
        for rr in range(rows):
            result[rr][c] = 1

    # Place 3s at diagonal neighbors of each original 1
    for r, c in ones:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] != 1:
                result[nr][nc] = 3

    # Place 2 at each original 1 position (highest priority)
    for r, c in ones:
        result[r][c] = 2

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/140c817e.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
