"""
Solver for ARC-AGI task c663677b.

The input is a 27x27 grid with a repeating tile pattern (period P in both
dimensions) that has been damaged: one or more rectangular regions have been
zeroed out. The output restores the original tiled pattern.

Algorithm:
  1. Detect the tile period P by finding the smallest shift where all
     non-zero cells are consistent with a periodic repetition.
  2. Build the P×P tile from the surviving (non-zero) cells.
  3. Tile the entire grid with that pattern.
"""

from __future__ import annotations
import json
import numpy as np
from typing import List


def _find_period(grid: np.ndarray, axis: int) -> int:
    """Find the smallest repeating period along `axis` (0=rows, 1=cols)."""
    if axis == 1:
        grid = grid.T
    H, W = grid.shape
    for p in range(1, H):
        ok = True
        covered = [False] * p
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 0:
                    covered[r % p] = True
                    # Check consistency with every earlier representative
                    for r2 in range(r % p, r, p):
                        if grid[r2, c] != 0 and grid[r2, c] != grid[r, c]:
                            ok = False
                            break
                    if not ok:
                        break
            if not ok:
                break
        if ok and all(covered):
            return p
    return H


def solve(grid: List[List[int]]) -> List[List[int]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    pr = _find_period(g, axis=0)
    pc = _find_period(g, axis=1)

    # Build tile from non-zero cells
    tile = np.zeros((pr, pc), dtype=int)
    for r in range(H):
        for c in range(W):
            if g[r, c] != 0:
                tile[r % pr, c % pc] = g[r, c]

    # Reconstruct full grid by tiling
    out = np.zeros_like(g)
    for r in range(H):
        for c in range(W):
            out[r, c] = tile[r % pr, c % pc]

    return out.tolist()


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/c663677b.json"
    )
    with open(path) as f:
        task = json.load(f)

    # Verify against training examples
    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        ok = result == expected
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    # Run on test examples
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        if "output" in ex:
            ok = result == ex["output"]
            print(f"Test  {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
        else:
            print(f"Test  {i}: solved (no ground truth to verify)")

    if all_pass:
        print("\nAll examples passed!")
