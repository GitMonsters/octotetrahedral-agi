from collections import Counter

def count_components(grid, color):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == color and (r, c) not in visited:
                count += 1
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == color:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
    return count

ORDER = [(0, 0), (2, 0), (1, 1), (2, 2), (0, 2), (1, 0), (2, 1), (0, 1), (1, 2)]

def transform(grid):
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    n_comp = count_components(grid, 2)
    result = [[bg] * 3 for _ in range(3)]
    for idx in range(min(n_comp, 9)):
        r, c = ORDER[idx]
        result[r][c] = 2
    return result
