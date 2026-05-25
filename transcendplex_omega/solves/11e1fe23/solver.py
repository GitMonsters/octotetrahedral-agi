def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find 3 colored (non-zero) pixels
    pts = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                pts.append((r, c, grid[r][c]))

    # Find the center: midpoint of the pair where all 3 points are
    # equidistant in Chebyshev distance
    center = None
    for i in range(3):
        for j in range(i + 1, 3):
            mr = (pts[i][0] + pts[j][0]) // 2
            mc = (pts[i][1] + pts[j][1]) // 2
            if (pts[i][0] + pts[j][0]) % 2 != 0 or (pts[i][1] + pts[j][1]) % 2 != 0:
                continue
            dists = [max(abs(mr - p[0]), abs(mc - p[1])) for p in pts]
            if dists[0] == dists[1] == dists[2]:
                center = (mr, mc)
                break
        if center:
            break

    # Place color 5 at center
    result[center[0]][center[1]] = 5

    # Place each original color 1 step from center toward its original position
    for r, c, color in pts:
        dr = 1 if r > center[0] else -1 if r < center[0] else 0
        dc = 1 if c > center[1] else -1 if c < center[1] else 0
        result[center[0] + dr][center[1] + dc] = color

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/11e1fe23.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
