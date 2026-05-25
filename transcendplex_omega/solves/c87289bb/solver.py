def solve(grid: list[list[int]]) -> list[list[int]]:
    """Extend pillar walls downward, relocating consumed pillars outside 2-bar segments."""
    rows = len(grid)
    cols = len(grid[0])

    # Find the 2-bar row (row containing 2s)
    two_bar_row = next(r for r in range(rows) if any(grid[r][c] == 2 for c in range(cols)))
    # The break row is directly above the 2-bar
    break_row = two_bar_row - 1

    # Find pillar columns from the top section
    pillar_cols = {c for c in range(cols) if grid[0][c] == 8}

    # Find contiguous 2-bar segments
    two_positions = [c for c in range(cols) if grid[two_bar_row][c] == 2]
    segments = []
    if two_positions:
        seg_start = seg_end = two_positions[0]
        for c in two_positions[1:]:
            if c == seg_end + 1:
                seg_end = c
            else:
                segments.append((seg_start, seg_end))
                seg_start = seg_end = c
        segments.append((seg_start, seg_end))

    # For each pillar consumed by a 2-bar, relocate it to just outside the nearest end
    consumed = set()
    new_walls: dict[int, int] = {}
    for seg_start, seg_end in segments:
        for p in pillar_cols:
            if seg_start <= p <= seg_end:
                consumed.add(p)
                left_dist = p - seg_start
                right_dist = seg_end - p
                if left_dist < right_dist:
                    new_walls[p] = seg_start - 1
                else:  # right closer or tie -> go right
                    new_walls[p] = seg_end + 1

    remaining = pillar_cols - consumed
    wall_cols = remaining | set(new_walls.values())

    # Build output
    output = [row[:] for row in grid]

    # Break row: wall columns + horizontal bars from old pillar pos to new pos
    break_8s = set(wall_cols)
    for old_pos, new_pos in new_walls.items():
        for c in range(min(old_pos, new_pos), max(old_pos, new_pos) + 1):
            break_8s.add(c)
    for c in break_8s:
        output[break_row][c] = 8

    # From 2-bar row to bottom: set wall columns to 8 (don't overwrite 2s)
    for r in range(two_bar_row, rows):
        for c in wall_cols:
            if output[r][c] == 0:
                output[r][c] = 8

    return output
