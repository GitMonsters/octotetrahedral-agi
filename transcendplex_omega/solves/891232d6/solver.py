def solve(grid: list[list[int]]) -> list[list[int]]:
    """Each 6 sends a path upward. At each horizontal bar of 7s, the path
    turns right: 8 marks the hit on the bar, 4/2/3 mark the corner below,
    and the path continues upward from one column past the bar's right end.
    If the turn is blocked by existing 7s, the path terminates with 6."""
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    sixes = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 6]

    for start_r, start_c in sixes:
        current_col = start_c
        r = start_r - 1

        while r >= 0:
            cell = grid[r][current_col]
            if cell == 7:
                # Find contiguous horizontal bar of 7s containing this cell
                bar_left = current_col
                bar_right = current_col
                while bar_left > 0 and grid[r][bar_left - 1] == 7:
                    bar_left -= 1
                while bar_right < cols - 1 and grid[r][bar_right + 1] == 7:
                    bar_right += 1

                exit_col = bar_right + 1
                turn_row = r + 1

                # Check if right turn is feasible
                feasible = exit_col < cols
                if feasible:
                    for c in range(current_col + 1, exit_col + 1):
                        if grid[turn_row][c] == 7:
                            feasible = False
                            break

                if feasible:
                    result[r][current_col] = 8
                    if not (turn_row == start_r and current_col == start_c):
                        result[turn_row][current_col] = 4
                    for c in range(current_col + 1, exit_col):
                        result[turn_row][c] = 2
                    result[turn_row][exit_col] = 3
                    result[r][exit_col] = 2
                    current_col = exit_col
                    r -= 1
                else:
                    if not (turn_row == start_r and current_col == start_c):
                        result[turn_row][current_col] = 6
                    break
            elif cell == 0:
                result[r][current_col] = 2
                r -= 1
            else:
                r -= 1

        if r < 0:
            result[0][current_col] = 6

    return result
