"""
arc_visualizer.py — ASCII visualizer for ARC task a32d8b75 grids.

Renders input/output grid pairs in the terminal using ANSI colour codes
(with automatic fallback to plain ASCII when the terminal does not support
colour).

Usage
-----
    # Visualize all training examples:
    python arc_visualizer.py

    # Visualize only training example 1:
    python arc_visualizer.py --example 1

    # Also show the test predictions:
    python arc_visualizer.py --test
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

Grid = List[List[int]]

_REPO_ROOT = Path(__file__).resolve().parent
_TASK_JSON = _REPO_ROOT / "arc-puzzle-catalog" / "dataset" / "tasks" / "a32d8b75.json"

# ── Colour palette (ANSI 256-colour approximations for ARC's 10 colours) ────
_ANSI_COLOURS = {
    0: "\033[48;5;0m",    # black
    1: "\033[48;5;21m",   # blue
    2: "\033[48;5;196m",  # red
    3: "\033[48;5;46m",   # green
    4: "\033[48;5;220m",  # yellow
    5: "\033[48;5;243m",  # grey
    6: "\033[48;5;201m",  # magenta
    7: "\033[48;5;208m",  # orange
    8: "\033[48;5;14m",   # light-blue / cyan
    9: "\033[48;5;160m",  # dark-red / maroon
}
_RESET = "\033[0m"
_SYMBOL = {i: str(i) for i in range(10)}

# Check whether the terminal supports ANSI colours
_USE_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _cell(value: int) -> str:
    sym = _SYMBOL.get(value, "?")
    if _USE_COLOUR:
        colour = _ANSI_COLOURS.get(value, "")
        return f"{colour} {sym} {_RESET}"
    return f" {sym}"


def render_grid(grid: Grid, title: str = "") -> None:
    """Print *grid* to stdout with an optional *title* header."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if title:
        print(f"\n  {title}  ({rows}×{cols})")
        print("  " + "─" * (cols * (4 if _USE_COLOUR else 2) + 2))
    for row in grid:
        print("  │" + "".join(_cell(v) for v in row) + "│")
    if title:
        print("  " + "─" * (cols * (4 if _USE_COLOUR else 2) + 2))


def render_pair(
    inp: Grid,
    out: Grid,
    label: str = "",
) -> None:
    """Render an input/output pair side-by-side (or stacked if too wide)."""
    print("\n" + "=" * 70)
    if label:
        print(f"  {label}")
    render_grid(inp, title="INPUT")
    render_grid(out, title="OUTPUT")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    show_test = "--test" in args
    example_idx: Optional[int] = None
    if "--example" in args:
        idx = args.index("--example") + 1
        if idx < len(args):
            example_idx = int(args[idx])

    from arc_task_a32d8b75_solver import solve, _load_task  # type: ignore

    data = _load_task(_TASK_JSON)

    for i, ex in enumerate(data["train"]):
        if example_idx is not None and i != example_idx:
            continue
        render_pair(ex["input"], ex["output"], label=f"Training example {i}")

    if show_test:
        for i, ex in enumerate(data["test"]):
            predicted = solve(ex["input"])
            render_pair(ex["input"], predicted, label=f"Test example {i} (predicted)")

    print()


if __name__ == "__main__":
    main()
