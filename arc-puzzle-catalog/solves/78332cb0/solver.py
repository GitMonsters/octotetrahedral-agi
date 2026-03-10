"""Solver for ARC-AGI task 78332cb0.

The input grid is divided into panels by rows/columns of 6.
Each panel's edges (top/bottom/left/right rows/cols) have 0 or 1 non-background
cells. Panels are chained so that the outgoing edge of one panel matches the
incoming edge of the next, forming a continuous path. The output re-arranges all
panels into a single row or column (with 6-separators) following that chain.
"""

from __future__ import annotations
import json
from typing import Optional


def solve(grid: list[list[int]]) -> list[list[int]]:
    panels = _extract_panels(grid)
    edges = [_get_edges(p) for p in panels]

    # Try vertical chain (top/bottom edges)
    chain = _build_chain(edges, start_key="top", end_key="bottom")
    if chain is not None:
        return _assemble_vertical([panels[i] for i in chain])

    # Try horizontal chain (left/right edges)
    chain = _build_chain(edges, start_key="left", end_key="right")
    if chain is not None:
        return _assemble_horizontal([panels[i] for i in chain])

    raise ValueError("Could not find a valid chain")


# ── panel extraction ──────────────────────────────────────────────


def _extract_panels(grid: list[list[int]]) -> list[list[list[int]]]:
    rows, cols = len(grid), len(grid[0])
    sep_rows = [r for r in range(rows) if all(grid[r][c] == 6 for c in range(cols))]
    sep_cols = [c for c in range(cols) if all(grid[r][c] == 6 for r in range(rows))]

    row_ranges = _ranges_from_separators(sep_rows, rows)
    col_ranges = _ranges_from_separators(sep_cols, cols)

    panels: list[list[list[int]]] = []
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            panels.append([row[c0:c1] for row in grid[r0:r1]])
    return panels


def _ranges_from_separators(seps: list[int], total: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    prev = 0
    for s in seps:
        if s > prev:
            ranges.append((prev, s))
        prev = s + 1
    if prev < total:
        ranges.append((prev, total))
    return ranges


# ── edge computation ──────────────────────────────────────────────


def _edge_val(cells: list[int]) -> Optional[tuple[int, ...]]:
    positions = tuple(i for i, c in enumerate(cells) if c != 7)
    return positions if positions else None


def _get_edges(panel: list[list[int]]) -> dict:
    h, w = len(panel), len(panel[0])
    return {
        "top": _edge_val(panel[0]),
        "bottom": _edge_val(panel[h - 1]),
        "left": _edge_val([panel[r][0] for r in range(h)]),
        "right": _edge_val([panel[r][w - 1] for r in range(h)]),
    }


# ── chain building (DFS) ─────────────────────────────────────────


def _build_chain(
    edges: list[dict],
    start_key: str,
    end_key: str,
) -> Optional[list[int]]:
    n = len(edges)
    # Start panel: the one with a blank incoming edge
    starts = [i for i in range(n) if edges[i][start_key] is None and edges[i][end_key] is not None]
    if len(starts) != 1:
        return None

    def dfs(current: int, path: list[int], used: set[int]) -> Optional[list[int]]:
        if len(path) == n:
            return list(path)
        link = edges[current][end_key]
        if link is None:
            return None  # chain ended too early
        for nxt in range(n):
            if nxt not in used and edges[nxt][start_key] == link:
                path.append(nxt)
                used.add(nxt)
                result = dfs(nxt, path, used)
                if result is not None:
                    return result
                path.pop()
                used.discard(nxt)
        return None

    start = starts[0]
    return dfs(start, [start], {start})


# ── assembly ──────────────────────────────────────────────────────


def _assemble_vertical(panels: list[list[list[int]]]) -> list[list[int]]:
    w = len(panels[0][0])
    result: list[list[int]] = []
    for i, panel in enumerate(panels):
        if i > 0:
            result.append([6] * w)
        result.extend(panel)
    return result


def _assemble_horizontal(panels: list[list[list[int]]]) -> list[list[int]]:
    h = len(panels[0])
    result: list[list[int]] = [[] for _ in range(h)]
    for i, panel in enumerate(panels):
        for r in range(h):
            if i > 0:
                result[r].append(6)
            result[r].extend(panel[r])
    return result


# ── validation harness ────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib, sys

    task_path = pathlib.Path(__file__).resolve().parents[2] / "dataset" / "tasks" / "78332cb0.json"
    with open(task_path) as f:
        data = json.load(f)

    all_pass = True
    for split in ("train", "test"):
        for i, pair in enumerate(data[split]):
            result = solve(pair["input"])
            ok = result == pair["output"]
            print(f"{split} pair {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
                print(f"  Expected {len(pair['output'])}x{len(pair['output'][0])}, got {len(result)}x{len(result[0])}")

    sys.exit(0 if all_pass else 1)
