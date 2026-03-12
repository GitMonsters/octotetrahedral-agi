def solve(grid: list[list[int]]) -> list[list[int]]:
    """Trace a path from the 2-source, detouring around 1/3 bars.
    
    Color 3 bars cause a RIGHT turn, color 1 bars cause a LEFT turn
    (relative to the direction of travel).
    """
    grid = [row[:] for row in grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find the 2 source
    src_r = src_c = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                src_r, src_c = r, c
                break
        else:
            continue
        break

    # Primary direction from edge position
    if src_c == 0:
        dr, dc = 0, 1
    elif src_c == cols - 1:
        dr, dc = 0, -1
    elif src_r == 0:
        dr, dc = 1, 0
    else:
        dr, dc = -1, 0

    r, c = src_r, src_c

    while 0 <= r < rows and 0 <= c < cols:
        grid[r][c] = 2

        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            break

        if grid[nr][nc] in (1, 3):
            bar_color = grid[nr][nc]

            if dc != 0:  # horizontal travel, vertical bar
                top = nr
                while top > 0 and grid[top - 1][nc] == bar_color:
                    top -= 1
                bottom = nr
                while bottom < rows - 1 and grid[bottom + 1][nc] == bar_color:
                    bottom += 1

                # 3→RIGHT turn, 1→LEFT turn relative to travel
                if (bar_color == 3 and dc > 0) or (bar_color == 1 and dc < 0):
                    target_r, step = bottom + 1, 1
                else:
                    target_r, step = top - 1, -1

                while r != target_r:
                    r += step
                    if 0 <= r < rows:
                        grid[r][c] = 2
                    else:
                        break

            else:  # vertical travel, horizontal bar
                left = nc
                while left > 0 and grid[nr][left - 1] == bar_color:
                    left -= 1
                right = nc
                while right < cols - 1 and grid[nr][right + 1] == bar_color:
                    right += 1

                if (bar_color == 3 and dr > 0) or (bar_color == 1 and dr < 0):
                    target_c, step = left - 1, -1
                else:
                    target_c, step = right + 1, 1

                while c != target_c:
                    c += step
                    if 0 <= c < cols:
                        grid[r][c] = 2
                    else:
                        break
        else:
            r, c = nr, nc

    return grid
