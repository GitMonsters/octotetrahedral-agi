"""Solver for ARC-AGI task b0722778.

Each row-group has three 2x2 blocks. Two share colors, two share pattern.
Output = unique colors mapped through the unique pattern.
"""
import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    H, W = len(grid), len(grid[0])
    sep_rows = {r for r in range(H) if all(grid[r][c] == 0 for c in range(W))}
    sep_cols = {c for c in range(W) if all(grid[r][c] == 0 for r in range(H))}

    def _bands(total, seps):
        bands, start = [], None
        for i in range(total):
            if i not in seps:
                if start is None: start = i
            else:
                if start is not None:
                    bands.append((start, i - 1)); start = None
        if start is not None: bands.append((start, total - 1))
        return bands

    col_bands = _bands(W, sep_cols)
    row_bands = _bands(H, sep_rows)

    def _pattern(block):
        flat = [v for row in block for v in row]
        m, key = {}, []
        for v in flat:
            if v not in m: m[v] = len(m)
            key.append(m[v])
        return tuple(key)

    def _colors(block):
        return frozenset(v for row in block for v in row)

    output = []
    for gi, (r0, r1) in enumerate(row_bands):
        blocks = [[grid[r][c0:c1+1] for r in range(r0, r1+1)] for c0, c1 in col_bands]
        pats = [_pattern(b) for b in blocks]
        cols = [_colors(b) for b in blocks]

        pp = next((i,j) for i in range(3) for j in range(i+1,3) if pats[i]==pats[j])
        cp = next((i,j) for i in range(3) for j in range(i+1,3) if cols[i]==cols[j])

        unique_pat = ({0,1,2} - set(pp)).pop()
        unique_col = ({0,1,2} - set(cp)).pop()
        bridge = ({0,1,2} - {unique_pat, unique_col}).pop()

        mapping = {}
        for r in range(len(blocks[bridge])):
            for c in range(len(blocks[bridge][0])):
                mapping[blocks[bridge][r][c]] = blocks[unique_col][r][c]

        result = [[mapping[blocks[unique_pat][r][c]] for c in range(len(blocks[unique_pat][0]))]
                  for r in range(len(blocks[unique_pat]))]

        for row in result: output.append(row)
        if gi < len(row_bands) - 1:
            output.append([0] * len(result[0]))

    return output
