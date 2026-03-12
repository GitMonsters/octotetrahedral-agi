import copy


def solve(grid):
    rows, cols_n = len(grid), len(grid[0])
    out = copy.deepcopy(grid)

    for r in range(rows):
        if grid[r][0] != 1:
            continue

        colored = sorted(c for c in range(cols_n) if grid[r][c] not in (0, 1))

        # Signal propagation: fill marker row left-to-right, stopping when
        # a colored cell has BOTH up and down 3-wide strips blocked by non-zero cells.
        signal_end = cols_n
        for cc in colored:
            up_clear = True
            down_clear = True
            for dc in [-1, 0, 1]:
                nc = cc + dc
                if 0 <= nc < cols_n:
                    if r - 1 >= 0 and grid[r - 1][nc] != 0:
                        up_clear = False
                    if r + 1 < rows and grid[r + 1][nc] != 0:
                        down_clear = False
            if not up_clear and not down_clear:
                signal_end = cc + 1
                break

        for c in range(signal_end):
            if out[r][c] == 0:
                out[r][c] = 1

        # Draw vertical 3-wide strips from each colored cell (left-to-right).
        # Relays from left strips can unblock right strips.
        relay_set = set()
        for cc in colored:
            for direction in [-1, 1]:
                _draw_strip(grid, out, r, cc, direction, rows, cols_n, relay_set)

    return out


def _draw_strip(grid, out, src_row, col, direction, rows, cols_n, relay_set):
    nr = src_row + direction
    if nr < 0 or nr >= rows:
        return

    left_c = col - 1
    right_c = col + 1

    # Left cell non-zero (and not a relay from a previous strip) blocks the strip
    if 0 <= left_c < cols_n and grid[nr][left_c] != 0 and (nr, left_c) not in relay_set:
        return

    # Draw 1s at 0-cells; register right-side non-zero cells as relays
    for dc in [-1, 0, 1]:
        nc = col + dc
        if 0 <= nc < cols_n:
            if out[nr][nc] == 0:
                out[nr][nc] = 1
            if grid[nr][nc] != 0 and nc >= col:
                relay_set.add((nr, nc))

    # If right cell is a non-zero relay, propagate from it
    if 0 <= right_c < cols_n and grid[nr][right_c] != 0:
        relay_col = right_c
        # Check if relay's vertical continuation is blocked
        nnr = nr + direction
        relay_blocked = False
        if nnr < 0 or nnr >= rows:
            relay_blocked = True
        else:
            left_of_relay = relay_col - 1
            if 0 <= left_of_relay < cols_n and grid[nnr][left_of_relay] != 0 \
               and (nnr, left_of_relay) not in relay_set:
                relay_blocked = True
        # Horizontal spread on same row only if vertical can continue
        if not relay_blocked:
            for dc in [-1, 0, 1]:
                nc = relay_col + dc
                if 0 <= nc < cols_n and out[nr][nc] == 0:
                    out[nr][nc] = 1
        # Continue vertical propagation
        _draw_strip(grid, out, nr, relay_col, direction, rows, cols_n, relay_set)
