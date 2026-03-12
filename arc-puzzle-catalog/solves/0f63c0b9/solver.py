def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all non-zero points
    points: list[tuple[int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                points.append((r, c, grid[r][c]))
    
    # Sort by row
    points.sort(key=lambda p: p[0])
    
    # Compute zone boundaries
    zones: list[tuple[int, int, int, int]] = []  # (zone_start, zone_end, point_row, color)
    for i, (pr, pc, color) in enumerate(points):
        if i == 0:
            zone_start = 0
        else:
            prev_row = points[i - 1][0]
            zone_start = (prev_row + pr) // 2 + 1
        if i == len(points) - 1:
            zone_end = rows - 1
        else:
            next_row = points[i + 1][0]
            zone_end = (pr + next_row) // 2
        zones.append((zone_start, zone_end, pr, color))
    
    # Build output
    result = [[0] * cols for _ in range(rows)]
    for zone_start, zone_end, point_row, color in zones:
        for r in range(zone_start, zone_end + 1):
            if r == point_row or r == 0 or r == rows - 1:
                # Full horizontal line
                for c in range(cols):
                    result[r][c] = color
            else:
                # Edges only
                result[r][0] = color
                result[r][cols - 1] = color
    
    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/0f63c0b9.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
