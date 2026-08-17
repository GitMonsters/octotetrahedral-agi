"""Grid serialization utilities for ARC tasks.

Grids are list[list[int]] with values 0-9. We render them in several
formats for the LLM (ASCII, Python nested list, and color stats), matching
the multi-format prompting used by top ARC program-synthesis systems.
"""

from __future__ import annotations

from typing import Iterable

Grid = list[list[int]]


def grid_dims(g: Grid) -> tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def colors_used(g: Grid) -> list[int]:
    return sorted({v for row in g for v in row})


def grid_ascii(g: Grid) -> str:
    """Render grid as an ASCII grid of single digits."""
    return "\n".join("".join(str(v) for v in row) for row in g)


def grid_python(g: Grid) -> str:
    """Render grid as a Python nested list literal."""
    return "[" + ",\n ".join("[" + ", ".join(str(v) for v in row) + "]" for row in g) + "]"


def grid_base64(g: Grid) -> str:
    """Render grid as a compact base64 row-major encoding."""
    import base64

    flat = bytes(v for row in g for v in row)
    return base64.b64encode(flat).decode()


def render_pair(index: int, input_g: Grid, output_g: Grid | None) -> str:
    """Render one train/test example in the multi-format prompt style."""
    lines = [f"### Example {index}"]
    lines.append(f"Input ({len(input_g)}x{len(input_g[0])}):")
    lines.append("```")
    lines.append(grid_ascii(input_g))
    lines.append("```")
    if output_g is not None:
        lines.append(f"Output ({len(output_g)}x{len(output_g[0])}):")
        lines.append("```")
        lines.append(grid_ascii(output_g))
        lines.append("```")
    return "\n".join(lines)


def render_task(task: dict, include_test: bool = False) -> str:
    """Render a full ARC task (train + optionally test inputs)."""
    parts = []
    for i, pair in enumerate(task["train"]):
        parts.append(render_pair(i, pair["input"], pair["output"]))
    if include_test:
        for i, pair in enumerate(task["test"]):
            parts.append(render_pair(len(task["train"]) + i, pair["input"], None))
    return "\n\n".join(parts)


def grids_equal(a: Grid, b: Grid) -> bool:
    return a == b


def cell_diffs(got: Grid, exp: Grid) -> tuple[int, int, int]:
    """Return (matching_cells, total_cells, mismatches) comparing two grids."""
    if not got or not exp:
        return (0, 0, 0)
    total = min(len(got), len(exp)) * min(len(got[0]), len(exp[0]))
    match = 0
    mismatches = []
    for r in range(max(len(got), len(exp))):
        for c in range(max(len(got[0]) if got else 0, len(exp[0]) if exp else 0)):
            got_v = got[r][c] if r < len(got) and c < len(got[0]) else None
            exp_v = exp[r][c] if r < len(exp) and c < len(exp[0]) else None
            if got_v == exp_v:
                match += 1
            elif len(mismatches) < 20:
                mismatches.append((r, c, exp_v, got_v))
    return (match, max(len(got), len(exp)) * max(len(got[0]) if got else 0, len(exp[0]) if exp else 0), mismatches)


def render_side_by_side(input_g: Grid, got: Grid, exp: Grid) -> str:
    """Render input | your output | expected as padded ASCII columns.

    The expected and got grids are shown under labeled columns so the model
    can visually diff the transformation at a glance.
    """
    def lines(g: Grid) -> list[str]:
        if not g:
            return [""]
        return ["".join(str(v) for v in row) for row in g]

    il, gl, el = lines(input_g), lines(got), lines(exp)
    max_h = max(len(il), len(gl), len(el), 1)
    max_w = max((len(r) for r in il + gl + el), default=0)
    col_w = max(max_w, 4)
    header = "INPUT".ljust(col_w) + " | " + "YOUR OUTPUT".ljust(col_w) + " | " + "EXPECTED"
    out = [header, "-" * len(header)]
    for r in range(max_h):
        i = il[r].ljust(col_w) if r < len(il) else " " * col_w
        g = gl[r].ljust(col_w) if r < len(gl) else " " * col_w
        e = el[r].ljust(col_w) if r < len(el) else " " * col_w
        out.append(f"{i} | {g} | {e}")
    return "\n".join(out)
