"""ARC puzzle 14754a24 solver.

Rule: Find plus/cross patterns (center + 4 cardinal neighbors) where all
in-bounds positions are yellow(4) or gray(5) and at least 2 are yellow.
Greedily select crosses with the most yellows first, claiming yellow cells.
Non-yellow (gray=5) positions in valid crosses become red(2).
"""


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])
    output = [row[:] for row in input_grid]

    # Find all candidate cross centers
    candidates = []
    for r in range(rows):
        for c in range(cols):
            positions = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            in_bounds = [
                (pr, pc)
                for pr, pc in positions
                if 0 <= pr < rows and 0 <= pc < cols
            ]
            if not all(input_grid[pr][pc] in (4, 5) for pr, pc in in_bounds):
                continue
            yellows = [
                (pr, pc) for pr, pc in in_bounds if input_grid[pr][pc] == 4
            ]
            if len(yellows) >= 2:
                candidates.append((len(yellows), r, c, in_bounds, yellows))

    # Greedy: process crosses with most yellows first
    candidates.sort(key=lambda x: -x[0])
    claimed: set[tuple[int, int]] = set()
    for _, r, c, in_bounds, yellows in candidates:
        unclaimed = [y for y in yellows if y not in claimed]
        if len(unclaimed) >= 2:
            for y in yellows:
                claimed.add(y)
            for pr, pc in in_bounds:
                if input_grid[pr][pc] == 5:
                    output[pr][pc] = 2

    return output
