from collections import Counter


def transform(grid):
    """For each 2x2 block of non-background color, add 4 corner bracket markers:
    - color 2 at (r-1, c-1)  (top-left diagonal)
    - color 9 at (r-1, c+2)  (top-right diagonal)
    - color 7 at (r+2, c-1)  (bottom-left diagonal)
    - color 0 at (r+2, c+2)  (bottom-right diagonal)
    """
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    color_count = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    bg = color_count.most_common(1)[0][0]

    other_colors = set(color_count.keys()) - {bg}
    if not other_colors:
        return result

    block_color = other_colors.pop()

    visited = set()
    blocks = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (r, c) not in visited:
                if (grid[r][c] == block_color and grid[r][c + 1] == block_color and
                        grid[r + 1][c] == block_color and grid[r + 1][c + 1] == block_color):
                    blocks.append((r, c))
                    visited.update([(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)])

    for r, c in blocks:
        markers = [
            (r - 1, c - 1, 2),
            (r - 1, c + 2, 9),
            (r + 2, c - 1, 7),
            (r + 2, c + 2, 0),
        ]
        for mr, mc, mval in markers:
            if 0 <= mr < rows and 0 <= mc < cols:
                result[mr][mc] = mval

    return result
