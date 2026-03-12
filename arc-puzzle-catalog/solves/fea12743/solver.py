#!/usr/bin/env python3
"""
Solver for ARC-AGI puzzle fea12743.

Rule: The 16×11 grid contains a 3×2 arrangement of 4×4 blocks (all drawn with 2s).
One block is the cell-wise OR (union) of two other blocks — it gets recolored to 3.
The two constituent blocks get recolored to 8. All other blocks stay as 2.
"""

import json
import copy
import sys
from itertools import combinations
from typing import List

Grid = List[List[int]]

BLOCK_ROWS = [(1, 5), (6, 10), (11, 15)]
BLOCK_COLS = [(1, 5), (6, 10)]


def extract_block(grid: Grid, ri: int, ci: int) -> tuple:
    r0, r1 = BLOCK_ROWS[ri]
    c0, c1 = BLOCK_COLS[ci]
    return tuple(tuple(grid[r][c0:c1]) for r in range(r0, r1))


def block_or(a: tuple, b: tuple) -> tuple:
    return tuple(
        tuple(max(a[r][c], b[r][c]) for c in range(len(a[0])))
        for r in range(len(a))
    )


def count_twos(block: tuple) -> int:
    return sum(v != 0 for row in block for v in row)


def solve(grid: Grid) -> Grid:
    out = copy.deepcopy(grid)
    positions = [(ri, ci) for ri in range(3) for ci in range(2)]
    blocks = {pos: extract_block(grid, *pos) for pos in positions}

    # Find the triple (a, b, c) where a OR b == c
    # c becomes 3, a and b become 8
    best_triple = None
    best_score = -1

    for a_pos, b_pos in combinations(positions, 2):
        or_block = block_or(blocks[a_pos], blocks[b_pos])
        for c_pos in positions:
            if c_pos == a_pos or c_pos == b_pos:
                continue
            if blocks[c_pos] == or_block:
                score = count_twos(or_block)
                # Tiebreaker: prefer pairs not sharing a row with composite
                same_row_penalty = sum(
                    1 for p in (a_pos, b_pos) if p[0] == c_pos[0]
                )
                adj_score = score * 10 - same_row_penalty
                if adj_score > best_score:
                    best_score = adj_score
                    best_triple = (a_pos, b_pos, c_pos)

    if best_triple is None:
        return out

    a_pos, b_pos, c_pos = best_triple

    # Recolor: composite → 3, constituents → 8
    color_map = {a_pos: 8, b_pos: 8, c_pos: 3}
    for pos, color in color_map.items():
        ri, ci = pos
        r0, r1 = BLOCK_ROWS[ri]
        c0, c1 = BLOCK_COLS[ci]
        for r in range(r0, r1):
            for c in range(c0, c1):
                if out[r][c] == 2:
                    out[r][c] = color

    return out


if __name__ == "__main__":
    task = json.load(
        open(
            "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/fea12743.json"
        )
    )

    passed = 0
    failed = 0

    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        if result == ex["output"]:
            print(f"✓ Train {i} PASS")
            passed += 1
        else:
            print(f"✗ Train {i} FAIL")
            failed += 1
            for r in range(len(result)):
                if result[r] != ex["output"][r]:
                    print(f"  Row {r}: got {result[r]}")
                    print(f"       exp {ex['output'][r]}")

    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        if "output" in ex and ex["output"]:
            if result == ex["output"]:
                print(f"✓ Test  {i} PASS")
                passed += 1
            else:
                print(f"✗ Test  {i} FAIL")
                failed += 1
        else:
            print(f"  Test  {i}: (no expected output to verify)")

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
