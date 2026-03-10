"""
Solver for ARC-AGI task b99e7126.

The 29x29 grid is a tiled pattern of 7x7 macro-cells (each 3x3) separated by
single-pixel borders. Some macro-cells are "stamped" with a modified pattern
that introduces a new color. The stamp's difference mask (positions where the
new color appears) is a 3x3 bitmap. The output places the stamp at every
macro-cell position indicated by that bitmap, anchored so the bitmap covers
all originally-stamped cells.
"""

from collections import Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])

    # Detect separator rows (uniform color) to find the tile period
    sep_rows = [r for r in range(H) if len(set(grid[r])) == 1]
    period = sep_rows[1] - sep_rows[0]
    cell_size = period - 1
    num_mr = (H - 1) // period
    num_mc = (W - 1) // period

    def get_cell(br: int, bc: int) -> tuple[tuple[int, ...], ...]:
        r0 = period * br + 1
        c0 = period * bc + 1
        return tuple(
            tuple(grid[r0 + dr][c0 + dc] for dc in range(cell_size))
            for dr in range(cell_size)
        )

    # Collect all macro-cell contents and find the base (most common) cell
    cells: dict[tuple[int, int], tuple] = {}
    counts: Counter = Counter()
    for br in range(num_mr):
        for bc in range(num_mc):
            cell = get_cell(br, bc)
            cells[(br, bc)] = cell
            counts[cell] += 1

    base = counts.most_common(1)[0][0]

    # Identify modified macro-cells and extract the stamp pattern
    modified = {(br, bc) for br in range(num_mr) for bc in range(num_mc) if cells[(br, bc)] != base}
    stamp = cells[next(iter(modified))]

    # The stamp color is the one present in the stamp but absent from the base
    base_colors = {c for row in base for c in row}
    stamp_colors = {c for row in stamp for c in row}
    new_color = (stamp_colors - base_colors).pop()

    # Build a binary mask: 1 where stamp has the new color
    mask = [
        [1 if stamp[dr][dc] == new_color else 0 for dc in range(cell_size)]
        for dr in range(cell_size)
    ]

    # Find the unique placement of the mask over the macro-grid that covers
    # all originally-modified cells
    for r_off in range(num_mr - cell_size + 1):
        for c_off in range(num_mc - cell_size + 1):
            mask_cells = {
                (r_off + dr, c_off + dc)
                for dr in range(cell_size)
                for dc in range(cell_size)
                if mask[dr][dc]
            }
            if modified.issubset(mask_cells):
                # Apply the stamp to every mask-indicated macro-cell
                output = [row[:] for row in grid]
                for br, bc in mask_cells:
                    r0 = period * br + 1
                    c0 = period * bc + 1
                    for dr in range(cell_size):
                        for dc in range(cell_size):
                            output[r0 + dr][c0 + dc] = stamp[dr][dc]
                return output

    return grid  # fallback


if __name__ == "__main__":
    import json, pathlib

    task_path = pathlib.Path(__file__).resolve().parents[2] / "dataset" / "tasks" / "b99e7126.json"
    with open(task_path) as f:
        task = json.load(f)

    for i, pair in enumerate(task["train"] + task.get("test", [])):
        label = f"train[{i}]" if i < len(task["train"]) else f"test[{i - len(task['train'])}]"
        result = solve(pair["input"])
        if "output" in pair:
            status = "PASS" if result == pair["output"] else "FAIL"
        else:
            status = "NO_EXPECTED"
        print(f"{label}: {status}")
