def transform(input_grid: list[list[int]]) -> list[list[int]]:
    from collections import Counter

    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]

    # Determine which edge has the pattern (most unique non-bg values)
    top_score = len(set(grid[0]) - {bg})
    bottom_score = len(set(grid[rows - 1]) - {bg})
    left_score = len(set(grid[r][0] for r in range(rows)) - {bg})
    right_score = len(set(grid[r][cols - 1] for r in range(rows)) - {bg})

    scores = {'top': top_score, 'bottom': bottom_score,
              'left': left_score, 'right': right_score}
    edge = max(scores, key=scores.get)

    if edge == 'top':
        # Horizontal pattern → reverse, tile downward from row 2
        pattern = [grid[0][c] for c in range(cols)][::-1]
        n = len(pattern)
        for k in range(rows - 2):
            color = pattern[k % n]
            grid[2 + k] = [color] * cols

    elif edge == 'bottom':
        # Horizontal pattern → reverse, tile upward from row rows-3
        pattern = [grid[rows - 1][c] for c in range(cols)][::-1]
        n = len(pattern)
        for k in range(rows - 2):
            color = pattern[k % n]
            grid[rows - 3 - k] = [color] * cols

    elif edge == 'left':
        # Vertical pattern → same order, tile rightward from col 2
        pattern = [grid[r][0] for r in range(rows)]
        n = len(pattern)
        for k in range(cols - 2):
            color = pattern[k % n]
            for r in range(rows):
                grid[r][2 + k] = color

    elif edge == 'right':
        # Vertical pattern → same order, tile leftward from col cols-3
        pattern = [grid[r][cols - 1] for r in range(rows)]
        n = len(pattern)
        for k in range(cols - 2):
            color = pattern[k % n]
            for r in range(rows):
                grid[r][cols - 3 - k] = color

    return grid
