"""
Solver for ARC-AGI task a32d8b75.

Task URL: https://arcprize.org/tasks/a32d8b75

Pattern
-------
The input grid is divided by vertical column(s) of colour 6 into a *key* section
(left, and optionally right) and a *puzzle* section (centre/right).

Key structure (width = N+2 columns, N is the stamp size):
  1. **Stamp section** – rows 0..(N+1): NxN pattern bordered by 0s, two colours.
  2. **Mask section**  – rows (N+2)..(bot_sep-1): shape drawn in a third colour.
  3. **Marker sections** (bordered by 6s):
       * Section with colour-4 cell  → determines rotation of mask onto puzzle.
       * Section with colour-7 cells → directional hint (informational).

Transformation:
  * The *inverted* stamp (swap the two stamp colours) tiles the puzzle grid,
    but only where the (possibly rotated) mask indicates.
  * **Single-key**: one key on the left, puzzle section on the right.
  * **Dual-key**:   keys on both sides, puzzle between them, with an embedded
    separator splitting the puzzle into left/right sub-grids.  Each key's mask
    controls tiling on the *opposite* sub-grid; stamps are also swapped.

Usage
-----
    python arc_task_a32d8b75_solver.py          # runs all examples & prints results
    python arc_task_a32d8b75_solver.py --json   # dumps test predictions as JSON
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

Grid = List[List[int]]

_REPO_ROOT = Path(__file__).resolve().parent
_TASK_JSON = _REPO_ROOT / "arc-puzzle-catalog" / "dataset" / "tasks" / "a32d8b75.json"


def _rotate_grid(grid: Grid, k: int) -> Grid:
    """Rotate *grid* by k*90° counter-clockwise (same convention as numpy.rot90)."""
    k = k % 4
    result: Grid = [row[:] for row in grid]
    for _ in range(k):
        r, c = len(result), len(result[0])
        result = [[result[row][c - 1 - col] for row in range(r)] for col in range(c)]
    return result


# ---------------------------------------------------------------------------
# Key extraction
# ---------------------------------------------------------------------------

def extract_key(
    grid: Grid,
    rows: int,
    col_start: int,
    col_end: int,
) -> Dict[str, Any]:
    """Parse one vertical key section and return its components."""
    key_width = col_end - col_start
    N = key_width - 2  # inner stamp dimension

    # ── Stamp ────────────────────────────────────────────────────────────────
    stamp_start: Optional[int] = None
    for r in range(N + 2):
        inner = [grid[r][col_start + 1 + c] for c in range(N)]
        if any(v != 0 for v in inner):
            if stamp_start is None:
                stamp_start = r

    if stamp_start is None:
        stamp_start = 1

    stamp: Grid = [grid[stamp_start + r][col_start + 1 : col_start + 1 + N] for r in range(N)]
    stamp_colors = sorted({c for row in stamp for c in row})
    inv_stamp: Optional[Grid] = None
    if len(stamp_colors) == 2:
        cA, cB = stamp_colors
        inv_stamp = [[cB if v == cA else cA for v in row] for row in stamp]

    # ── Bottom key separator (first all-6 row inside key columns) ────────────
    bot_sep_row: Optional[int] = None
    for r in range(N + 2, rows):
        if all(grid[r][c] == 6 for c in range(col_start, col_end)):
            bot_sep_row = r
            break

    # ── Marker sections ──────────────────────────────────────────────────────
    marker4_pos: Optional[Tuple[int, int]] = None

    if bot_sep_row is not None:
        def _inner_start(r_start: int) -> int:
            for c in range(col_start, col_end):
                if grid[r_start][c] != 6:
                    return c
            return col_start + 1

        bs1_row = bot_sep_row + 1
        bs1_cs = _inner_start(bs1_row)
        bs1 = [grid[bs1_row + r][bs1_cs : bs1_cs + 3] for r in range(3)]

        bs2_sep: Optional[int] = None
        for r in range(bs1_row + 3, rows):
            if all(grid[r][c] == 6 for c in range(col_start, col_end)):
                bs2_sep = r
                break

        for r in range(3):
            for c in range(3):
                if bs1[r][c] == 4:
                    marker4_pos = (r, c)

    # ── Rotation from 4-marker position ──────────────────────────────────────
    rot_k = 0
    if marker4_pos is not None:
        r4, c4 = marker4_pos
        if r4 == 0 and c4 == 0:
            rot_k = 0
        elif r4 == 0 and c4 == 2:
            rot_k = 3
        elif r4 == 2 and c4 == 0:
            rot_k = 1
        elif r4 == 2 and c4 == 2:
            rot_k = 2

    # ── Mask (binarised bounding-box crop) ───────────────────────────────────
    mid_start = N + 2
    mid_end = bot_sep_row if bot_sep_row is not None else rows
    mask_full = [grid[r][col_start:col_end] for r in range(mid_start, mid_end)]

    mrc = len(mask_full)
    r_min, r_max, c_min, c_max = mrc, -1, key_width, -1
    for r in range(mrc):
        for c in range(key_width):
            if mask_full[r][c] != 0:
                if r < r_min:
                    r_min = r
                if r > r_max:
                    r_max = r
                if c < c_min:
                    c_min = c
                if c > c_max:
                    c_max = c

    mask_core: Optional[Grid] = None
    if r_max >= 0:
        mask_core = [
            [1 if mask_full[r][c] != 0 else 0 for c in range(c_min, c_max + 1)]
            for r in range(r_min, r_max + 1)
        ]

    return {
        "N": N,
        "stamp": stamp,
        "inv_stamp": inv_stamp,
        "marker4_pos": marker4_pos,
        "rot_k": rot_k,
        "mask_core": mask_core,
    }


# ---------------------------------------------------------------------------
# Tile application
# ---------------------------------------------------------------------------

def apply_tile(
    output: Grid,
    mask: Optional[Grid],
    inv_stamp: Optional[Grid],
    N: int,
    rot_k: int,
    marker4_pos: Optional[Tuple[int, int]],
    out_rows: int,
    out_cols: int,
    invert_anchor: bool = False,
) -> None:
    """Paint the inverted stamp tile onto *output* wherever *mask* is active."""
    if mask is None or inv_stamp is None:
        return

    rotated = _rotate_grid(mask, rot_k)
    rot_rows = len(rotated)
    rot_cols = len(rotated[0])

    if marker4_pos is not None:
        r4, c4 = marker4_pos
        if invert_anchor:
            r4, c4 = 2 - r4, 2 - c4
        start_r = 0 if r4 == 0 else out_rows - rot_rows * N
        start_c = 0 if c4 == 0 else out_cols - rot_cols * N
    else:
        start_r, start_c = 0, 0

    dr = (-start_r) % N
    dc = (-start_c) % N

    for r in range(out_rows):
        for c in range(out_cols):
            cell_r = r - start_r
            cell_c = c - start_c
            if cell_r < 0 or cell_c < 0:
                continue
            mr, mc = cell_r // N, cell_c // N
            if mr < 0 or mr >= rot_rows or mc < 0 or mc >= rot_cols:
                continue
            if rotated[mr][mc]:
                output[r][c] = inv_stamp[(r + dr) % N][(c + dc) % N]


# ---------------------------------------------------------------------------
# Main solve function
# ---------------------------------------------------------------------------

def solve(grid: Grid) -> Grid:
    """Transform one ARC task a32d8b75 input grid and return the output grid."""
    rows = len(grid)
    cols = len(grid[0])

    # Locate vertical separator columns (entirely colour 6)
    sep_cols = [c for c in range(cols) if all(grid[r][c] == 6 for r in range(rows))]
    if not sep_cols:
        return [row[:] for row in grid]

    first_sep = sep_cols[0]
    last_sep = sep_cols[-1]
    dual_key = len(sep_cols) > 1

    if not dual_key:
        # ── Single-key mode ──────────────────────────────────────────────────
        left_key = extract_key(grid, rows, 0, first_sep)
        N = left_key["N"]
        right_start = first_sep + 1
        out_cols = cols - right_start
        output: Grid = [row[right_start:] for row in grid]

        apply_tile(
            output,
            left_key["mask_core"],
            left_key["inv_stamp"],
            N,
            left_key["rot_k"],
            left_key["marker4_pos"],
            rows,
            out_cols,
        )
        return output

    # ── Dual-key mode ────────────────────────────────────────────────────────
    left_key = extract_key(grid, rows, 0, first_sep)
    right_key = extract_key(grid, rows, last_sep + 1, cols)
    N = left_key["N"]

    puzzle_start = first_sep + 1
    puzzle_end = last_sep
    puzzle_cols = puzzle_end - puzzle_start
    output = [row[puzzle_start:puzzle_end] for row in grid]

    # Locate the embedded separator block inside the puzzle section
    embed_width = N + 2
    embed_start: Optional[int] = None
    for cs in range(puzzle_cols - embed_width + 1):
        is_uniform = all(
            len({output[r][c] for r in range(rows)}) == 1
            for c in range(cs, cs + embed_width)
        )
        if is_uniform:
            embed_start = cs
            break

    if embed_start is None:
        return output  # fallback

    embed_end = embed_start + embed_width
    right_sub_start = embed_end

    # Left sub-grid: mask from right key, stamp inverted right, rotation from left key
    left_sub: Grid = [row[:embed_start] for row in output]
    apply_tile(
        left_sub,
        right_key["mask_core"],
        right_key["inv_stamp"],
        N,
        left_key["rot_k"],
        left_key["marker4_pos"],
        rows,
        embed_start,
        invert_anchor=True,
    )

    # Right sub-grid: mask from left key, stamp inverted left, rotation from right key
    right_sub: Grid = [row[right_sub_start:] for row in output]
    apply_tile(
        right_sub,
        left_key["mask_core"],
        left_key["inv_stamp"],
        N,
        right_key["rot_k"],
        right_key["marker4_pos"],
        rows,
        puzzle_cols - right_sub_start,
        invert_anchor=True,
    )

    # Reassemble
    for r in range(rows):
        output[r] = left_sub[r] + output[r][embed_start:embed_end] + right_sub[r]

    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_task(path: Path = _TASK_JSON) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def _run_training(data: Dict[str, Any]) -> bool:
    all_pass = True
    for i, ex in enumerate(data["train"]):
        actual = solve(ex["input"])
        expected = ex["output"]
        if actual == expected:
            print(f"Train {i}: PASS  ({len(actual)}×{len(actual[0])})")
        else:
            all_pass = False
            diffs = sum(
                1
                for r in range(len(expected))
                for c in range(len(expected[0]))
                if r >= len(actual)
                or c >= len(actual[0])
                or actual[r][c] != expected[r][c]
            )
            print(f"Train {i}: FAIL  ({diffs} cell diffs)")
    return all_pass


def _run_test(data: Dict[str, Any], emit_json: bool = False) -> None:
    predictions = []
    for i, ex in enumerate(data["test"]):
        result = solve(ex["input"])
        predictions.append(result)
        print(f"Test  {i}: solved  ({len(result)}×{len(result[0])})")

    if emit_json:
        out = json.dumps(predictions, indent=2)
        print("\n--- test predictions (JSON) ---")
        print(out)


def main() -> None:
    emit_json = "--json" in sys.argv
    data = _load_task()

    print("=" * 60)
    print("ARC task a32d8b75 — training validation")
    print("=" * 60)
    all_pass = _run_training(data)

    print()
    print("=" * 60)
    print("ARC task a32d8b75 — test predictions")
    print("=" * 60)
    _run_test(data, emit_json=emit_json)

    print()
    if all_pass:
        print("✓ All training examples passed.")
    else:
        print("✗ Some training examples failed.")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
