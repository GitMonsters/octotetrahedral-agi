from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    """For each connected component of non-zero cells containing a colored marker
    (value != 0 and != 1), recolor every '1' cell whose 8 neighbors are all non-zero
    with the marker's color."""
    H, W = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    visited = [[False] * W for _ in range(H)]

    def bfs(sr: int, sc: int) -> list[tuple[int, int]]:
        q = deque([(sr, sc)])
        visited[sr][sc] = True
        cells = [(sr, sc)]
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] != 0:
                    visited[nr][nc] = True
                    q.append((nr, nc))
                    cells.append((nr, nc))
        return cells

    for r in range(H):
        for c in range(W):
            if not visited[r][c] and grid[r][c] != 0:
                comp = bfs(r, c)
                marker = None
                for cr, cc in comp:
                    if grid[cr][cc] not in (0, 1):
                        marker = grid[cr][cc]
                        break
                if marker is None:
                    continue
                for cr, cc in comp:
                    if grid[cr][cc] != 1:
                        continue
                    if all(
                        0 <= cr + dr < H and 0 <= cc + dc < W and grid[cr + dr][cc + dc] != 0
                        for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1)
                        if not (dr == 0 and dc == 0)
                    ):
                        result[cr][cc] = marker

    return result


if __name__ == "__main__":
    import sys, json

    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        expected = ex.get('output')
        if expected:
            status = "PASS" if result == expected else "FAIL"
            print(f"Example {i}: {status}")
        else:
            print(f"Example {i}: no expected output")
