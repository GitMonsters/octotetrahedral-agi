def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Flood fill to find exterior 0-cells (connected to grid border)
    exterior = [[False] * cols for _ in range(rows)]
    queue = []
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid[r][c] == 0:
                if not exterior[r][c]:
                    exterior[r][c] = True
                    queue.append((r, c))
    i = 0
    while i < len(queue):
        r, c = queue[i]
        i += 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and not exterior[nr][nc]:
                exterior[nr][nc] = True
                queue.append((nr, nc))

    # For each non-zero cell, detect L-corners and classify them
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                continue

            # Collect 4-connected non-zero neighbor directions
            nbrs = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 0:
                    nbrs.append((dr, dc))

            if len(nbrs) != 2:
                continue

            (dr1, dc1), (dr2, dc2) = nbrs
            if dr1 * dr2 + dc1 * dc2 != 0:  # not perpendicular
                continue

            # L-corner: check diagonal in the direction of both neighbors
            diag_r, diag_c = r + dr1 + dr2, c + dc1 + dc2

            if 0 <= diag_r < rows and 0 <= diag_c < cols:
                if grid[diag_r][diag_c] == 0 and not exterior[diag_r][diag_c]:
                    result[r][c] = 4  # outer (convex) corner
                else:
                    result[r][c] = 2  # inner (concave) corner
            else:
                result[r][c] = 2

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/15663ba9.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
