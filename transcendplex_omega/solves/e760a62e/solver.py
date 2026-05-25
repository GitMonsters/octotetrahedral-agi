from collections import defaultdict


def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find divider rows/cols (entirely 8s)
    div_rows = {r for r in range(rows) if all(grid[r][c] == 8 for c in range(cols))}
    div_cols = {c for c in range(cols) if all(grid[r][c] == 8 for r in range(rows))}

    # Compute block ranges
    def get_ranges(divs, total):
        ranges = []
        start = None
        for i in range(total):
            if i in divs:
                if start is not None:
                    ranges.append((start, i - 1))
                    start = None
            elif start is None:
                start = i
        if start is not None:
            ranges.append((start, total - 1))
        return ranges

    br_ranges = get_ranges(div_rows, rows)
    bc_ranges = get_ranges(div_cols, cols)
    n_br, n_bc = len(br_ranges), len(bc_ranges)

    # Find colored markers per block
    markers = {}
    for bi, (rs, re) in enumerate(br_ranges):
        for bj, (cs, ce) in enumerate(bc_ranges):
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    if grid[r][c] not in (0, 8):
                        markers[(bi, bj)] = grid[r][c]

    # For each color, determine row/col fills
    block_fill = [[set() for _ in range(n_bc)] for _ in range(n_br)]

    for color in set(markers.values()):
        pts = [(br, bc) for (br, bc), v in markers.items() if v == color]

        by_row = defaultdict(list)
        by_col = defaultdict(list)
        for br, bc in pts:
            by_row[br].append(bc)
            by_col[bc].append(br)

        for br, bcs in by_row.items():
            if len(bcs) >= 2:
                for bc in range(min(bcs), max(bcs) + 1):
                    block_fill[br][bc].add(color)

        for bc, brs in by_col.items():
            if len(brs) >= 2:
                for br in range(min(brs), max(brs) + 1):
                    block_fill[br][bc].add(color)

    # Build output
    output = [row[:] for row in grid]
    for bi, (rs, re) in enumerate(br_ranges):
        for bj, (cs, ce) in enumerate(bc_ranges):
            fills = block_fill[bi][bj]
            if not fills:
                c = 0
            elif len(fills) == 1:
                c = next(iter(fills))
            else:
                c = 6  # intersection of 2 and 3
            for r in range(rs, re + 1):
                for col in range(cs, ce + 1):
                    output[r][col] = c
    return output
