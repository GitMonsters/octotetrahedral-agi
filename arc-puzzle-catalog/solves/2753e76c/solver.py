def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]

    color_counts: dict[int, int] = {}

    def flood_fill(r: int, c: int, color: int) -> None:
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if 0 <= cr < rows and 0 <= cc < cols and not visited[cr][cc] and grid[cr][cc] == color:
                visited[cr][cc] = True
                stack.extend([(cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)])

    # Count distinct connected rectangles per color
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                flood_fill(r, c, color)
                color_counts[color] = color_counts.get(color, 0) + 1

    # Sort colors by rectangle count descending
    sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])

    n_colors = len(sorted_colors)
    max_count = sorted_colors[0][1] if sorted_colors else 0

    # Build staircase: each row right-filled with the color's count
    result = [[0] * max_count for _ in range(n_colors)]
    for i, (color, count) in enumerate(sorted_colors):
        for j in range(max_count - count, max_count):
            result[i][j] = color

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/2753e76c.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
