from collections import Counter


def transform(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])

    # Count colors: two most common are backgrounds, rest are markers
    color_counts = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    sorted_colors = color_counts.most_common()
    backgrounds = {sorted_colors[0][0], sorted_colors[1][0]}
    markers = {c for c, _ in sorted_colors[2:]}

    # Map each marker to its surrounding background via majority vote
    neighbor_bg_counts: dict[int, Counter] = {m: Counter() for m in markers}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in markers:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] in backgrounds:
                        neighbor_bg_counts[grid[r][c]][grid[nr][nc]] += 1
    marker_to_bg: dict[int, int] = {}
    for m in markers:
        if neighbor_bg_counts[m]:
            marker_to_bg[m] = neighbor_bg_counts[m].most_common(1)[0][0]

    bg_to_marker = {v: k for k, v in marker_to_bg.items()}

    # Determine region background for every cell
    cell_bg = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            cell_bg[r][c] = grid[r][c] if grid[r][c] in backgrounds else marker_to_bg[grid[r][c]]

    # Trace 4 diagonal rays from each marker, painting the region's marker color
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in markers:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < rows and 0 <= nc < cols:
                        bg = cell_bg[nr][nc]
                        if bg in bg_to_marker:
                            result[nr][nc] = bg_to_marker[bg]
                        nr += dr
                        nc += dc

    return result
