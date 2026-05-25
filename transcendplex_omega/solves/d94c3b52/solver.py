def solve(grid: list[list[int]]) -> list[list[int]]:
    """Tiles matching the 8-tile pattern become 8; tiles between matching pairs (same row/col) become 7."""
    rows, cols = len(grid), len(grid[0])

    # Find separator rows/cols (all zeros)
    sep_rows = [r for r in range(rows) if all(grid[r][c] == 0 for c in range(cols))]
    sep_cols = [c for c in range(cols) if all(grid[r][c] == 0 for r in range(rows))]

    # Tile row/col ranges
    tile_rows = [(sep_rows[i] + 1, sep_rows[i + 1])
                 for i in range(len(sep_rows) - 1) if sep_rows[i + 1] - sep_rows[i] > 1]
    tile_cols = [(sep_cols[i] + 1, sep_cols[i + 1])
                 for i in range(len(sep_cols) - 1) if sep_cols[i + 1] - sep_cols[i] > 1]

    n_tr, n_tc = len(tile_rows), len(tile_cols)

    def get_structure(tr: int, tc: int) -> tuple[tuple[int, ...], ...]:
        rs, re = tile_rows[tr]
        cs, ce = tile_cols[tc]
        return tuple(tuple(1 if grid[r][c] != 0 else 0 for c in range(cs, ce))
                     for r in range(rs, re))

    # Find the 8-tile and its structure
    eight_struct = None
    for tr in range(n_tr):
        for tc in range(n_tc):
            rs, re = tile_rows[tr]
            cs, ce = tile_cols[tc]
            if any(grid[r][c] == 8 for r in range(rs, re) for c in range(cs, ce)):
                eight_struct = get_structure(tr, tc)
                break
        if eight_struct:
            break

    # All tiles matching the 8-tile structure
    matching = {(tr, tc) for tr in range(n_tr) for tc in range(n_tc)
                if get_structure(tr, tc) == eight_struct}

    # Tiles between matching pairs in same row or column
    seven = set()
    for tr in range(n_tr):
        rm = sorted(tc for r, tc in matching if r == tr)
        if len(rm) >= 2:
            seven |= {(tr, tc) for tc in range(rm[0] + 1, rm[-1]) if (tr, tc) not in matching}
    for tc in range(n_tc):
        cm = sorted(r for r, c in matching if c == tc)
        if len(cm) >= 2:
            seven |= {(tr, tc) for tr in range(cm[0] + 1, cm[-1]) if (tr, tc) not in matching}

    # Build output
    result = [row[:] for row in grid]
    for tr, tc in matching:
        rs, re = tile_rows[tr]
        cs, ce = tile_cols[tc]
        for r in range(rs, re):
            for c in range(cs, ce):
                if result[r][c] == 1:
                    result[r][c] = 8

    for tr, tc in seven:
        rs, re = tile_rows[tr]
        cs, ce = tile_cols[tc]
        for r in range(rs, re):
            for c in range(cs, ce):
                if result[r][c] == 1:
                    result[r][c] = 7

    return result
