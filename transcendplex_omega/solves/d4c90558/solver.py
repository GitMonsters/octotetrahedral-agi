def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find all unique frame colors (non-zero, non-5)
    colors = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in (0, 5):
                colors.add(grid[r][c])

    results = []
    for color in colors:
        # Find bounding box of this color
        min_r = min_c = float('inf')
        max_r = max_c = -1
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    min_r, max_r = min(min_r, r), max(max_r, r)
                    min_c, max_c = min(min_c, c), max(max_c, c)

        # Count 5s inside the bounding box
        count = sum(
            1 for r in range(min_r, max_r + 1)
            for c in range(min_c, max_c + 1)
            if grid[r][c] == 5
        )
        results.append((count, color))

    results.sort()
    max_count = max(cnt for cnt, _ in results)

    return [[color] * cnt + [0] * (max_count - cnt) for cnt, color in results]
