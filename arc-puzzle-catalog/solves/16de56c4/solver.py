import json, sys
from collections import Counter, deque, defaultdict


def solve(grid):
    rows, cols = len(grid), len(grid[0])

    def analyze_line(cells):
        """Check if a line has a valid pattern: one color with 2+ regularly-spaced
        cells and at most one singleton of a different color."""
        if len(cells) < 2:
            return None

        color_groups = defaultdict(list)
        for pos, color in cells:
            color_groups[color].append(pos)

        pattern_color = None
        pattern_positions = None

        for color in sorted(color_groups.keys()):
            positions = sorted(color_groups[color])
            if len(positions) >= 2:
                diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
                if len(set(diffs)) == 1:
                    if pattern_color is not None:
                        return None  # multiple pattern colors → invalid
                    pattern_color = color
                    pattern_positions = positions

        if pattern_color is None:
            return None

        singletons = []
        for color in sorted(color_groups.keys()):
            if color != pattern_color:
                for pos in sorted(color_groups[color]):
                    singletons.append((pos, color))

        if len(singletons) > 1:
            return None

        return {
            "pattern_color": pattern_color,
            "pattern_positions": pattern_positions,
            "spacing": pattern_positions[1] - pattern_positions[0],
            "singletons": singletons,
        }

    # Collect valid patterns for each direction
    row_infos = {}
    for r in range(rows):
        cells = [(c, grid[r][c]) for c in range(cols) if grid[r][c] != 0]
        info = analyze_line(cells)
        if info:
            row_infos[r] = info

    col_infos = {}
    for c in range(cols):
        cells = [(r, grid[r][c]) for r in range(rows) if grid[r][c] != 0]
        info = analyze_line(cells)
        if info:
            col_infos[c] = info

    output = [row[:] for row in grid]

    def apply_pattern(line_type, idx, info, line_length):
        spacing = info["spacing"]
        start = info["pattern_positions"][0]
        pattern_color = info["pattern_color"]
        singletons = info["singletons"]

        # All positions on the pattern grid within the line
        offset = start % spacing
        all_pattern_pos = set()
        p = offset
        while p < line_length:
            all_pattern_pos.add(p)
            p += spacing

        def set_cell(pos, color):
            if line_type == "row":
                output[idx][pos] = color
            else:
                output[pos][idx] = color

        if not singletons:
            # No singleton: fill all pattern positions with pattern color
            for p in all_pattern_pos:
                set_cell(p, pattern_color)
        else:
            s_pos, s_color = singletons[0]
            if s_pos in all_pattern_pos:
                # On-pattern singleton: fill from min to max of all cells
                all_cells = info["pattern_positions"] + [s_pos]
                lo, hi = min(all_cells), max(all_cells)
                for p in all_pattern_pos:
                    if lo <= p <= hi:
                        set_cell(p, s_color)
            else:
                # Off-pattern singleton: extend pattern fully, keep singleton
                for p in all_pattern_pos:
                    set_cell(p, pattern_color)

    # Process the direction with more valid patterns
    if len(row_infos) >= len(col_infos):
        for r, info in row_infos.items():
            apply_pattern("row", r, info, cols)
    else:
        for c, info in col_infos.items():
            apply_pattern("col", c, info, rows)

    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/16de56c4.json") as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Train {i}: PASS ✓")
        else:
            mismatches = sum(
                1
                for r in range(len(expected))
                for c in range(len(expected[0]))
                if result[r][c] != expected[r][c]
            )
            print(f"Train {i}: FAIL - {mismatches} mismatches")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
            all_pass = False

    for i, ex in enumerate(data["test"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Test {i}: PASS ✓")
        else:
            mismatches = sum(
                1
                for r in range(len(expected))
                for c in range(len(expected[0]))
                if result[r][c] != expected[r][c]
            )
            print(f"Test {i}: FAIL - {mismatches} mismatches")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
            all_pass = False

    if all_pass:
        print("\nAll examples pass! ✓")
