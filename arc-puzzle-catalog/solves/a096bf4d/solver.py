import copy
from collections import Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find separator rows/cols (all 0s)
    sep_rows = [r for r in range(rows) if all(grid[r][c] == 0 for c in range(cols))]
    sep_cols = [c for c in range(cols) if all(grid[r][c] == 0 for r in range(rows))]

    # Extract block row/col ranges
    block_row_ranges = []
    for i in range(len(sep_rows) - 1):
        s, e = sep_rows[i] + 1, sep_rows[i + 1]
        if e > s:
            block_row_ranges.append((s, e))
    block_col_ranges = []
    for i in range(len(sep_cols) - 1):
        s, e = sep_cols[i] + 1, sep_cols[i + 1]
        if e > s:
            block_col_ranges.append((s, e))

    # Extract 2x2 interior of each block (offset 1 from block edges)
    block_interiors = {}
    for br, (rs, re) in enumerate(block_row_ranges):
        for bc, (cs, ce) in enumerate(block_col_ranges):
            interior = tuple(
                tuple(grid[r][c] for c in range(cs + 1, ce - 1))
                for r in range(rs + 1, re - 1)
            )
            block_interiors[(br, bc)] = interior

    # Most common interior is the base pattern
    base = Counter(block_interiors.values()).most_common(1)[0][0]

    # Find deviations: group by (value, rel_row, rel_col)
    deviations: dict[tuple, list[tuple]] = {}
    for (br, bc), interior in block_interiors.items():
        for ri in range(len(base)):
            for ci in range(len(base[0])):
                if interior[ri][ci] != base[ri][ci]:
                    deviations.setdefault((interior[ri][ci], ri, ci), []).append((br, bc))

    # For each deviation group, fill lines between pairs sharing a row or column
    fills: dict[tuple, int] = {}
    for (val, ri, ci), blocks in deviations.items():
        filled = set(blocks)
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                br1, bc1 = blocks[i]
                br2, bc2 = blocks[j]
                if br1 == br2:
                    for bc in range(min(bc1, bc2), max(bc1, bc2) + 1):
                        filled.add((br1, bc))
                elif bc1 == bc2:
                    for br in range(min(br1, br2), max(br1, br2) + 1):
                        filled.add((br, bc1))
        for br, bc in filled:
            fills[(br, bc, ri, ci)] = val

    # Apply fills to output
    output = copy.deepcopy(grid)
    for (br, bc, ri, ci), val in fills.items():
        rs = block_row_ranges[br][0]
        cs = block_col_ranges[bc][0]
        output[rs + 1 + ri][cs + 1 + ci] = val

    return output
