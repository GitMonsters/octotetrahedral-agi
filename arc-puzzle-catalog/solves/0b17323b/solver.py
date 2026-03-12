def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    
    # Find all blue (1) dots
    ones = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                ones.append((r, c))
    
    # Sort by row then col
    ones.sort()
    
    # Compute step from first two points
    dr = ones[1][0] - ones[0][0]
    dc = ones[1][1] - ones[0][1]
    
    # Continue from last blue dot
    last_r, last_c = ones[-1]
    r, c = last_r + dr, last_c + dc
    while 0 <= r < rows and 0 <= c < cols:
        result[r][c] = 2
        r += dr
        c += dc
    
    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/0b17323b.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
