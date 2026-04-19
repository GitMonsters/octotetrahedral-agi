from collections import Counter


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    R = len(input_grid)
    C = len(input_grid[0])

    # Find background color (most common)
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    def count_components(target_color):
        visited = [[False] * C for _ in range(R)]
        count = 0
        for r in range(R):
            for c in range(C):
                if input_grid[r][c] == target_color and not visited[r][c]:
                    count += 1
                    queue = [(r, c)]
                    visited[r][c] = True
                    while queue:
                        cr, cc = queue.pop(0)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and input_grid[nr][nc] == target_color:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
        return count

    def count_non_bg_components():
        visited = [[False] * C for _ in range(R)]
        count = 0
        for r in range(R):
            for c in range(C):
                if input_grid[r][c] != bg and not visited[r][c]:
                    color = input_grid[r][c]
                    count += 1
                    queue = [(r, c)]
                    visited[r][c] = True
                    while queue:
                        cr, cc = queue.pop(0)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and input_grid[nr][nc] == color:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
        return count

    # Determine level based on block count
    if bg == 8:
        level = count_non_bg_components()
    else:
        n_eight = count_components(8)
        level = max(1, n_eight - 1) if n_eight > 0 else 0

    # Mark positions added at each level
    marks = set()
    if level >= 1:
        marks.add((2, 0))
    if level >= 2:
        marks.update([(0, 0), (1, 1)])
    if level >= 3:
        marks.add((2, 2))
    if level >= 4:
        marks.add((0, 2))
    if level >= 5:
        marks.update([(1, 0), (2, 1)])
    if level >= 6:
        marks.update([(0, 1), (1, 2)])

    # Build 3x3 output
    output = [[bg] * 3 for _ in range(3)]
    for r, c in marks:
        output[r][c] = 1

    return output
