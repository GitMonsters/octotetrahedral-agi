def solve(grid: list[list[int]]) -> list[list[int]]:
    """From the 3x3 block of 8s, emit 4 diagonal lines of 2s from corners and
    4 clockwise-rotating staircases of 4s from each face.

    Staircases: TOP→right, RIGHT→down, BOTTOM→left, LEFT→up.
    Each staircase alternates single-4 and 3-bar steps, stopping when a 5 blocks.
    Diagonal 2-lines stop at grid edge, a 5 in the path, or a 5-wall across the step.
    """
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find top-left of 8-block
    tr = tc = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and tr is None:
                tr, tc = r, c
    cr, cc = tr + 1, tc + 1  # center of 3x3

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    # 4 staircases of 4s (one per face, rotating clockwise)
    for main_dr, main_dc, step_dr, step_dc in [
        (-1, 0, 0, 1),   # TOP face → step right
        (0, 1, 1, 0),    # RIGHT face → step down
        (1, 0, 0, -1),   # BOTTOM face → step left
        (0, -1, -1, 0),  # LEFT face → step up
    ]:
        k = 0
        while True:
            base_r = cr + (k + 2) * main_dr
            base_c = cc + (k + 2) * main_dc
            step_offset = k

            if k % 2 == 0:
                r = base_r + step_offset * step_dr
                c = base_c + step_offset * step_dc
                if not in_bounds(r, c) or grid[r][c] == 5:
                    break
                result[r][c] = 4
            else:
                stopped = False
                for i in (-1, 0, 1):  # inner to outer
                    so = step_offset + i
                    r = base_r + so * step_dr
                    c = base_c + so * step_dc
                    if not in_bounds(r, c) or grid[r][c] == 5:
                        stopped = True
                        break
                    result[r][c] = 4
                if stopped:
                    break
            k += 1

    # 4 diagonal lines of 2s from each corner of the 8-block
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        r, c = cr + dr, cc + dc  # start at corner
        while True:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc) or grid[nr][nc] == 5:
                break
            # Stop if both cells flanking the diagonal step are 5 ("wall")
            if (in_bounds(nr, c) and grid[nr][c] == 5 and
                    in_bounds(r, nc) and grid[r][nc] == 5):
                break
            r, c = nr, nc
            if result[r][c] == 0:
                result[r][c] = 2

    return result
