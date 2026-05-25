"""
Solver for ARC-AGI task 93b4f4b3

Pattern: The grid has a template (left) with a border color and holes (0s),
and colored patterns (right). Each colored pattern's shape matches the holes
in one template section. Fill each section's holes with the matching color.
"""

from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows = len(grid)
    cols = len(grid[0])
    border_color = grid[0][0]

    # Find template width: contiguous border_color cells in row 0
    template_width = 0
    for c in range(cols):
        if grid[0][c] == border_color:
            template_width = c + 1
        else:
            break

    pattern_start = template_width

    # Find border rows (all border_color in template columns)
    border_rows = []
    for r in range(rows):
        if all(grid[r][c] == border_color for c in range(template_width)):
            border_rows.append(r)

    # Sections are between consecutive border rows
    sections = []
    for i in range(len(border_rows) - 1):
        r1, r2 = border_rows[i], border_rows[i + 1]
        if r2 - r1 > 1:
            sections.append((r1, r2))

    # Extract hole positions per section (relative to section interior)
    section_holes: list[set[tuple[int, int]]] = []
    for r_start, r_end in sections:
        holes = set()
        for r in range(r_start + 1, r_end):
            for c in range(template_width):
                if grid[r][c] == 0:
                    holes.add((r - r_start - 1, c))
        section_holes.append(holes)

    # Extract colored patterns from the right side, per section
    color_patterns: dict[int, set[tuple[int, int]]] = {}
    for r_start, r_end in sections:
        for r in range(r_start + 1, r_end):
            for c in range(pattern_start, cols):
                val = grid[r][c]
                if val != 0:
                    if val not in color_patterns:
                        color_patterns[val] = set()
                    color_patterns[val].add((r - r_start - 1, c - pattern_start))

    # Build output: copy template, then fill holes with matching colors
    output = [list(grid[r][:template_width]) for r in range(rows)]

    for si, holes in enumerate(section_holes):
        r_start, _ = sections[si]
        for color, positions in color_patterns.items():
            if positions == holes:
                for ro, co in holes:
                    output[r_start + 1 + ro][co] = color
                break

    return output


if __name__ == "__main__":
    import json
    import sys

    path = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/93b4f4b3.json"
    task = json.load(open(path))

    all_pass = True
    for split in ["train", "test"]:
        for i, pair in enumerate(task[split]):
            result = solve(pair["input"])
            if "output" in pair:
                if result == pair["output"]:
                    print(f"{split} {i}: PASS ✓")
                else:
                    print(f"{split} {i}: FAIL ✗")
                    all_pass = False
                    for r, (got, exp) in enumerate(zip(result, pair["output"])):
                        if got != exp:
                            print(f"  row {r}: got {got}, expected {exp}")
            else:
                print(f"{split} {i}: (no expected output)")
                print(json.dumps(result))

    sys.exit(0 if all_pass else 1)
