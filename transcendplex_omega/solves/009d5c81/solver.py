def solve(grid: list[list[int]]) -> list[list[int]]:
    """Replace 8s with a color determined by the shape of the 1-pattern; remove 1s.

    Shape mapping (normalized 3x3 patterns):
      Plus shape (5 cells)    → color 2
      Down-arrow (6 cells)    → color 3
      Up-arrow (6 cells)      → color 7
    """
    rows, cols = len(grid), len(grid[0])

    # Extract 1-cell positions and normalize
    ones = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    min_r = min(r for r, c in ones)
    min_c = min(c for r, c in ones)
    pattern = frozenset((r - min_r, c - min_c) for r, c in ones)

    # Known pattern → replacement color lookup
    PLUS = frozenset([(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)])
    DOWN_ARROW = frozenset([(0, 0), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)])
    UP_ARROW = frozenset([(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1)])

    pattern_to_color = {
        PLUS: 2,
        DOWN_ARROW: 3,
        UP_ARROW: 7,
    }

    color = pattern_to_color[pattern]

    # Build output: replace 8→color, 1→0, rest unchanged
    result = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                result[r][c] = color
            elif grid[r][c] == 1:
                result[r][c] = 0
            else:
                result[r][c] = grid[r][c]
    return result


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        expected = ex.get('output')
        if expected:
            status = "PASS" if result == expected else "FAIL"
            print(f"Example {i}: {status}")
        else:
            print(f"Example {i}: no expected output")
