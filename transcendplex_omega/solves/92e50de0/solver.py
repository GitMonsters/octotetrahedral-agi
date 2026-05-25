"""Solver for ARC-AGI task 92e50de0.

Pattern: Grid is divided into blocks by colored separator lines.
One block contains a colored pattern. The pattern is tiled to every
block sharing the same (row_parity, col_parity) as the source block.
All other blocks are cleared to 0.
"""

import json
import copy
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    H = len(grid)
    W = len(grid[0])

    # Find separator color: first row that is entirely one non-zero color
    sep_color = None
    for r in range(H):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != 0:
            sep_color = grid[r][0]
            break

    # Find separator row/col positions
    sep_rows = [r for r in range(H) if all(grid[r][c] == sep_color for c in range(W))]
    sep_cols = [c for c in range(W) if all(grid[r][c] == sep_color for r in range(H))]

    # Build cell ranges from separator positions
    def cell_ranges(seps, size):
        ranges = []
        start = 0
        for s in seps:
            if s > start:
                ranges.append((start, s))
            start = s + 1
        if start < size:
            ranges.append((start, size))
        return ranges

    row_ranges = cell_ranges(sep_rows, H)
    col_ranges = cell_ranges(sep_cols, W)

    # Find the block containing the pattern
    pattern_block = None
    pattern_content = None
    for bi, (rs, re) in enumerate(row_ranges):
        for bj, (cs, ce) in enumerate(col_ranges):
            for r in range(rs, re):
                for c in range(cs, ce):
                    if grid[r][c] != 0 and grid[r][c] != sep_color:
                        # Extract full cell content
                        pattern_content = [
                            [grid[r2][c2] for c2 in range(cs, ce)]
                            for r2 in range(rs, re)
                        ]
                        pattern_block = (bi, bj)
                        break
                if pattern_block:
                    break
            if pattern_block:
                break
        if pattern_block:
            break

    src_rp = pattern_block[0] % 2
    src_cp = pattern_block[1] % 2
    ph = len(pattern_content)
    pw = len(pattern_content[0])

    output = copy.deepcopy(grid)

    for bi, (rs, re) in enumerate(row_ranges):
        for bj, (cs, ce) in enumerate(col_ranges):
            if bi % 2 == src_rp and bj % 2 == src_cp:
                # Tile the pattern (clipped to cell size)
                for dr in range(min(ph, re - rs)):
                    for dc in range(min(pw, ce - cs)):
                        output[rs + dr][cs + dc] = pattern_content[dr][dc]
            else:
                # Clear cell
                for r in range(rs, re):
                    for c in range(cs, ce):
                        output[r][c] = 0

    return output


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/92e50de0.json"))

    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        match = result == pair["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(result)):
                if result[r] != pair["output"][r]:
                    print(f"  Row {r} diff: got {result[r]}")
                    print(f"             exp {pair['output'][r]}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            match = result == pair["output"]
            print(f"Test  {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
        else:
            print(f"Test  {i}: produced {len(result)}x{len(result[0])} output")

    print(f"\nAll passed: {all_pass}")
