def _find_all_4_blocks(grid):
    height, width = len(grid), len(grid[0])
    blocks = set()
    for row in range(height - 1):
        for col in range(width - 1):
            if grid[row][col] == grid[row][col + 1] == grid[row + 1][col] == grid[row + 1][col + 1] == 4:
                blocks.add((row, col))
    return blocks


def _find_clusters(blocks):
    clusters = []
    seen = set()
    for block in blocks:
        if block in seen:
            continue
        stack = [block]
        seen.add(block)
        cluster = []
        while stack:
            row, col = stack.pop()
            cluster.append((row, col))
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in blocks and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        clusters.append(cluster)
    return clusters


def transform(grid):
    blocks = _find_all_4_blocks(grid)
    if not blocks:
        return [row[:] for row in grid]

    clusters = _find_clusters(blocks)
    selected = set()

    if all(len(cluster) == 1 for cluster in clusters):
        for cluster in clusters:
            selected.update(cluster)
    else:
        for cluster in clusters:
            if len(cluster) >= 3:
                selected.update(cluster)
            elif len(cluster) == 2:
                selected.add(max(cluster, key=lambda cell: (cell[1], cell[0])))

    output = [row[:] for row in grid]
    for row, col in selected:
        for drow in (0, 1):
            for dcol in (0, 1):
                output[row + drow][col + dcol] = 8
    return output
