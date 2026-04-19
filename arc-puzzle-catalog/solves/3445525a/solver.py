def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    Each 2x2 foreground square gets 4 single-pixel diagonal corner markers:
      top-left  (r-1, c-1) = 5 (gray)
      top-right (r-1, c+2) = 3 (green)
      bot-left  (r+2, c-1) = 7 (orange)
      bot-right (r+2, c+2) = 9 (maroon)
    """
    import numpy as np
    from collections import Counter

    grid = np.array(grid)
    output = grid.copy()
    rows, cols = grid.shape

    # Detect background (most common) and foreground colors
    counts = Counter(grid.flatten().tolist())
    bg = counts.most_common(1)[0][0]
    fg_colors = [c for c, _ in counts.most_common() if c != bg]

    if not fg_colors:
        return output.tolist()

    fg = fg_colors[0]

    # Find top-left corners of all 2x2 foreground squares
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (grid[r, c] == fg and grid[r, c + 1] == fg
                    and grid[r + 1, c] == fg and grid[r + 1, c + 1] == fg):
                # Skip if this cell is interior to a larger block
                if r > 0 and grid[r - 1, c] == fg:
                    continue
                if c > 0 and grid[r, c - 1] == fg:
                    continue

                # Place the 4 corner markers
                if r - 1 >= 0 and c - 1 >= 0:
                    output[r - 1, c - 1] = 5
                if r - 1 >= 0 and c + 2 < cols:
                    output[r - 1, c + 2] = 3
                if r + 2 < rows and c - 1 >= 0:
                    output[r + 2, c - 1] = 7
                if r + 2 < rows and c + 2 < cols:
                    output[r + 2, c + 2] = 9

    return output.tolist()
