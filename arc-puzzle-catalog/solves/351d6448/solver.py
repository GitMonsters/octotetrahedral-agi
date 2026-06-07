def solve(grid: list[list[int]]) -> list[list[int]]:
    # Rule: the grid contains N blocks of equal height separated by rows of a
    # constant value (5).  The output is the next block in the sequence.
    #
    # For single-content-row blocks (Train 0/1):
    #   - Detect whether the pattern shifts right or grows, and extrapolate by
    #     one more step.
    #
    # For multi-row content blocks (test):
    #   - There are two non-zero colours: an "advancing" colour (starts on the
    #     left) and a "static" colour (fills the rest).
    #   - The bottom row's frontier (rightmost advancing cell) advances by 1
    #     each block.
    #   - Upper rows' frontiers cascade: frontier_r = rightmost non-zero
    #     template position <= frontier_{r+1}.
    #   - Build the next block using these frontiers.

    rows, cols = len(grid), len(grid[0])

    # find separator value (a row that is entirely one non-zero value)
    sep_val = None
    for row in grid:
        vals = set(row)
        if len(vals) == 1 and row[0] != 0:
            sep_val = row[0]
            break

    # split into blocks
    blocks: list[list[list[int]]] = []
    cur: list[list[int]] = []
    for row in grid:
        if sep_val is not None and all(v == sep_val for v in row):
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(row[:])
    if cur:
        blocks.append(cur)

    n_blocks = len(blocks)
    block_h = len(blocks[0])

    # count content rows per block
    def content_row_indices(block: list[list[int]]) -> list[int]:
        return [i for i, row in enumerate(block) if any(v != 0 for v in row)]

    content_indices = content_row_indices(blocks[-1])

    # ── CASE A: single content row per block (shift or grow) ────────────────
    if len(content_indices) == 1:
        content_ri = content_indices[0]
        patterns = [b[content_ri][:] for b in blocks]

        def bounds(row: list[int]) -> tuple[int, int]:
            left = next((i for i, v in enumerate(row) if v != 0), -1)
            right = next((i for i, v in enumerate(reversed(row)) if v != 0), -1)
            if left == -1:
                return -1, -1
            return left, len(row) - 1 - right

        lefts = [bounds(p)[0] for p in patterns]
        rights = [bounds(p)[1] for p in patterns]
        left_diffs = [lefts[i + 1] - lefts[i] for i in range(n_blocks - 1)]
        right_diffs = [rights[i + 1] - rights[i] for i in range(n_blocks - 1)]

        last = patterns[-1]
        l_last, r_last = bounds(last)

        # Shift: both endpoints move by the same constant non-zero amount
        if (left_diffs and right_diffs
                and all(d == left_diffs[0] for d in left_diffs)
                and all(d == right_diffs[0] for d in right_diffs)
                and left_diffs[0] == right_diffs[0]
                and left_diffs[0] != 0):
            shift = left_diffs[0]
            new_row = [0] * cols
            for i in range(l_last, r_last + 1):
                new_pos = i + shift
                if 0 <= new_pos < cols:
                    new_row[new_pos] = last[i]

        # Grow: right endpoint advances, left stays fixed
        elif (right_diffs
              and all(d == right_diffs[0] for d in right_diffs)
              and right_diffs[0] > 0):
            growth = right_diffs[0]
            new_row = last[:]
            extend_val = last[r_last]
            for i in range(r_last + 1, min(r_last + growth + 1, cols)):
                new_row[i] = extend_val

        else:
            new_row = last[:]

        result = []
        for i in range(block_h):
            result.append(new_row if i == content_ri else [0] * cols)
        return result

    # ── CASE B: multi-row content (staircase advancing/static pattern) ──────

    # Build non-zero template (union across all blocks)
    template_nz: list[list[int]] = []
    for ri in range(block_h):
        nz = [c for c in range(cols) if any(b[ri][c] != 0 for b in blocks)]
        template_nz.append(nz)

    # Driver row: the one with the most non-zero template positions
    driver_row = max(range(block_h), key=lambda r: len(template_nz[r]))

    # Advancing colour: the leftmost non-zero value in the driver row of B1
    advancing: int = next(v for v in blocks[0][driver_row] if v != 0)

    # Static colour: any other non-zero, non-separator value
    static: int | None = None
    for b in blocks:
        for row in b:
            for v in row:
                if v != 0 and v != sep_val and v != advancing:
                    static = v
                    break
            if static is not None:
                break
        if static is not None:
            break

    def frontier(block: list[list[int]], row: int) -> int:
        """Rightmost column in `row` that carries the advancing colour."""
        for c in range(cols - 1, -1, -1):
            if block[row][c] == advancing:
                return c
        return -1

    driver_fronts = [frontier(b, driver_row) for b in blocks]
    deltas = [driver_fronts[i + 1] - driver_fronts[i]
              for i in range(n_blocks - 1)]
    delta = next((d for d in reversed(deltas) if d != 0), 1)

    new_driver_front = driver_fronts[-1] + delta

    # Cascade frontiers from driver row upward
    new_fronts = [-1] * block_h
    new_fronts[driver_row] = new_driver_front
    for ri in range(driver_row - 1, -1, -1):
        valid = [c for c in template_nz[ri] if c <= new_fronts[ri + 1]]
        new_fronts[ri] = max(valid) if valid else -1

    # Build the output block
    result = []
    for ri in range(block_h):
        row_out = [0] * cols
        front = new_fronts[ri]
        for c in template_nz[ri]:
            if c <= front:
                row_out[c] = advancing
            elif static is not None:
                row_out[c] = static
        result.append(row_out)

    return result
