from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]

    type_a = []  # stem above bar → value 3 (left half)
    type_b = []  # stem below bar → value 1 (right half)

    def bfs(r: int, c: int) -> list[tuple[int, int]]:
        q = deque([(r, c)])
        visited[r][c] = True
        cells = [(r, c)]
        while q:
            cr, cc = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 2:
                    visited[nr][nc] = True
                    q.append((nr, nc))
                    cells.append((nr, nc))
        return cells

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2 and not visited[r][c]:
                cells = bfs(r, c)
                cells_set = set(cells)
                classified = False
                # Try horizontal bar
                for cr, cc in cells:
                    if (cr, cc + 1) in cells_set and (cr, cc + 2) in cells_set:
                        center_col = cc + 1
                        stem = [x for x in cells if x not in {(cr, cc), (cr, cc + 1), (cr, cc + 2)}]
                        if len(stem) == 1:
                            sr, sc = stem[0]
                            if sr == cr - 1 and sc == center_col:
                                type_a.append((cr, cc))
                                classified = True
                            elif sr == cr + 1 and sc == center_col:
                                type_b.append((cr, cc))
                                classified = True
                    if classified:
                        break

    type_a.sort()
    type_b.sort()

    output = [[0] * 6 for _ in range(3)]

    # Column-first fill order within each 2×2 corner grid
    pos_a = [(0, 0), (2, 0), (0, 2), (2, 2)]
    pos_b = [(0, 3), (2, 3), (0, 5), (2, 5)]

    for i in range(min(len(type_a), 4)):
        r, c = pos_a[i]
        output[r][c] = 3

    for i in range(min(len(type_b), 4)):
        r, c = pos_b[i]
        output[r][c] = 1

    return output
