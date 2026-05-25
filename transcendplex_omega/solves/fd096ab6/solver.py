"""
ARC-AGI puzzle fd096ab6 solver.

Pattern: One color acts as a "template" shape (the one with the most cells).
All other colored objects are incomplete subsets of that template shape.
Each incomplete object is completed to match the full template, preserving its
anchor position. Colors with cells in multiple separate locations get each
group completed independently.
"""

import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows = len(grid)
    cols = len(grid[0])
    bg = 1

    # Collect cells by color
    color_cells: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                color_cells.setdefault(grid[r][c], []).append((r, c))

    if not color_cells:
        return [row[:] for row in grid]

    # Template = color with the most cells
    template_color = max(color_cells, key=lambda k: len(color_cells[k]))
    template_cells = color_cells[template_color]

    # Compute template offsets from top-left of bounding box
    min_r = min(r for r, c in template_cells)
    min_c = min(c for r, c in template_cells)
    template_offsets = set((r - min_r, c - min_c) for r, c in template_cells)

    result = [row[:] for row in grid]

    for color, cells in color_cells.items():
        if color == template_color:
            continue

        assigned: set[tuple[int, int]] = set()

        for cr, cc in cells:
            if (cr, cc) in assigned:
                continue

            best_anchor = None
            best_count = 0

            # Try mapping this cell to each template offset
            for off_r, off_c in template_offsets:
                ar, ac = cr - off_r, cc - off_c

                count = sum(
                    1
                    for cr2, cc2 in cells
                    if (cr2, cc2) not in assigned
                    and (cr2 - ar, cc2 - ac) in template_offsets
                )

                if count > best_count:
                    best_count = count
                    best_anchor = (ar, ac)

            if best_anchor:
                ar, ac = best_anchor
                for cr2, cc2 in cells:
                    if (cr2 - ar, cc2 - ac) in template_offsets:
                        assigned.add((cr2, cc2))

                for off_r, off_c in template_offsets:
                    nr, nc = ar + off_r, ac + off_c
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr][nc] = color

    return result


if __name__ == "__main__":
    with open("/tmp/arc_task_fd096ab6.json") as f:
        task = json.load(f)

    all_pass = True
    for i, example in enumerate(task["train"]):
        output = solve(example["input"])
        expected = example["output"]
        if output == expected:
            print(f"Training example {i}: PASS")
        else:
            all_pass = False
            print(f"Training example {i}: FAIL")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if output[r][c] != expected[r][c]:
                        print(f"  Mismatch at ({r},{c}): got {output[r][c]}, expected {expected[r][c]}")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")

    if "test" in task:
        for i, test in enumerate(task["test"]):
            result = solve(test["input"])
            if "output" in test:
                if result == test["output"]:
                    print(f"Test {i}: PASS")
                else:
                    print(f"Test {i}: FAIL")
                    for r in range(len(test["output"])):
                        for c in range(len(test["output"][0])):
                            if result[r][c] != test["output"][r][c]:
                                print(f"  Mismatch at ({r},{c}): got {result[r][c]}, expected {test['output'][r][c]}")
