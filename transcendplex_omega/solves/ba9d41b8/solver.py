import copy


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)
    visited = [[False] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                # Find rectangle extent
                r2 = r
                while r2 + 1 < rows and grid[r2 + 1][c] == color:
                    r2 += 1
                c2 = c
                while c2 + 1 < cols and grid[r][c2 + 1] == color:
                    c2 += 1

                for rr in range(r, r2 + 1):
                    for cc in range(c, c2 + 1):
                        visited[rr][cc] = True

                # Checkerboard interior: border stays, inside alternates
                for rr in range(r + 1, r2):
                    for cc in range(c + 1, c2):
                        if (rr - r + cc - c) % 2 == 0:
                            result[rr][cc] = 0

    return result
