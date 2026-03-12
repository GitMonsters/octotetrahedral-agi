from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])

    # Find the two non-zero colors
    colors = sorted(set(v for row in grid for v in row if v != 0))

    def count_interior_zeros(color):
        """Count zeros enclosed within the boundary of a colored shape."""
        # Find bounding box
        positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        min_r = min(r for r, c in positions)
        max_r = max(r for r, c in positions)
        min_c = min(c for r, c in positions)
        max_c = max(c for r, c in positions)

        h = max_r - min_r + 1
        w = max_c - min_c + 1

        # Extract sub-grid and find exterior zeros via flood fill
        visited = [[False] * w for _ in range(h)]
        queue = deque()

        # Seed flood fill from border cells that aren't this color
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                    if grid[min_r + r][min_c + c] != color and not visited[r][c]:
                        visited[r][c] = True
                        queue.append((r, c))

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                    if grid[min_r + nr][min_c + nc] != color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))

        # Count interior zeros: zeros inside bounding box not reached by flood fill
        count = 0
        for r in range(h):
            for c in range(w):
                if grid[min_r + r][min_c + c] == 0 and not visited[r][c]:
                    count += 1
        return count, min_c

    results = []
    for color in colors:
        count, left_pos = count_interior_zeros(color)
        results.append((left_pos, color, count))

    # Sort by leftmost column position
    results.sort()

    # Build 3x3 output: each color starts on a new row, fills left-to-right,
    # continues to next rows if needed, pads remainder of last row with 0.
    output = [[0] * 3 for _ in range(3)]
    row = 0
    for _, color, count in results:
        if row >= 3:
            break
        remaining = count
        while remaining > 0 and row < 3:
            can_place = min(remaining, 3)
            for c in range(can_place):
                output[row][c] = color
            remaining -= can_place
            row += 1

    return output
