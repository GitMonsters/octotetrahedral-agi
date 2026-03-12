def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find wall of 5s (a full row or full column)
    wall_type = None  # 'row' or 'col'
    wall_pos = None
    for r in range(rows):
        if all(grid[r][c] == 5 for c in range(cols)):
            wall_type = 'row'
            wall_pos = r
            break
    if wall_type is None:
        for c in range(cols):
            if all(grid[r][c] == 5 for r in range(rows)):
                wall_type = 'col'
                wall_pos = c
                break

    # Find colored segments (parallel to wall)
    segments = []  # (color, distance, r_start, r_end, c_start, c_end) of rectangle

    if wall_type == 'row':
        # Segments are horizontal lines
        for r in range(rows):
            if r == wall_pos:
                continue
            c = 0
            while c < cols:
                if grid[r][c] not in (0, 5):
                    color = grid[r][c]
                    start = c
                    while c < cols and grid[r][c] == color:
                        c += 1
                    end = c - 1
                    # Extend toward wall
                    if wall_pos > r:
                        r_start, r_end = r, wall_pos - 1
                    else:
                        r_start, r_end = wall_pos + 1, r
                    distance = abs(r - wall_pos)
                    segments.append((distance, color, r_start, r_end, start, end))
                else:
                    c += 1
    else:
        # Segments are vertical lines
        for c in range(cols):
            if c == wall_pos:
                continue
            r = 0
            while r < rows:
                if grid[r][c] not in (0, 5):
                    color = grid[r][c]
                    start = r
                    while r < rows and grid[r][c] == color:
                        r += 1
                    end = r - 1
                    # Extend toward wall
                    if wall_pos > c:
                        c_start, c_end = c, wall_pos - 1
                    else:
                        c_start, c_end = wall_pos + 1, c
                    distance = abs(c - wall_pos)
                    segments.append((distance, color, start, end, c_start, c_end))
                else:
                    r += 1

    # Paint farthest first so closest overwrites (closer = higher priority)
    segments.sort(key=lambda x: -x[0])
    for dist, color, r_start, r_end, c_start, c_end in segments:
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                result[r][c] = color

    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/13713586.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
