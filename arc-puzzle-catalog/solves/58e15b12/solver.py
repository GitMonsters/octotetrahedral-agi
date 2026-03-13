def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Two pairs of parallel vertical rails (colors 8 and 3) extend outward
    in both directions (up and down). Every H rows (original height of the
    rail pair), each rail spreads 1 column further apart. Where rails of
    different colors overlap, the cell becomes color 6.
    """
    rows = len(grid)
    cols = len(grid[0])

    rail_pairs = _find_rail_pairs(grid, rows, cols)
    output = [[0] * cols for _ in range(rows)]

    for color, r_start, H, lc, rc in rail_pairs:
        max_steps = rows + cols
        for step in range(max_steps + 1):
            new_lc = lc - step
            new_rc = rc + step

            if new_lc < 0 and new_rc >= cols:
                break

            if step == 0:
                for r in range(r_start, min(r_start + H, rows)):
                    _place(output, r, new_lc, color, cols)
                    _place(output, r, new_rc, color, cols)
            else:
                # Upward block
                up_start = r_start - step * H
                up_end = r_start - (step - 1) * H - 1
                for r in range(max(0, up_start), min(rows, up_end + 1)):
                    _place(output, r, new_lc, color, cols)
                    _place(output, r, new_rc, color, cols)

                # Downward block
                down_start = r_start + H + (step - 1) * H
                down_end = r_start + H + step * H - 1
                for r in range(max(0, down_start), min(rows, down_end + 1)):
                    _place(output, r, new_lc, color, cols)
                    _place(output, r, new_rc, color, cols)

    return output


def _place(output, r, c, color, cols):
    if 0 <= c < cols:
        if output[r][c] == 0:
            output[r][c] = color
        elif output[r][c] != color:
            output[r][c] = 6


def _find_rail_pairs(grid, rows, cols):
    found_colors = set()
    rail_pairs = []
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            if color != 0 and color not in found_colors:
                found_colors.add(color)
                cells = [(rr, cc) for rr in range(rows) for cc in range(cols)
                         if grid[rr][cc] == color]
                min_r = min(r for r, _ in cells)
                max_r = max(r for r, _ in cells)
                col_vals = sorted(set(c for _, c in cells))
                rail_pairs.append((color, min_r, max_r - min_r + 1,
                                   col_vals[0], col_vals[1]))
    return rail_pairs
