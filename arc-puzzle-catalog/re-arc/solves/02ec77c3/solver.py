def transform(input_grid):
    """
    Rule: Non-background pixels appear only on left (col 0) and right (col W-1) edges.
    For rows where either edge is non-background:
      - Left half (cols 0..mid-1) fills with left edge color
      - Center column (mid = W//2) fills with 4 (Yellow separator)
      - Right half (cols mid+1..W-1) fills with right edge color
    Background rows stay unchanged.
    """
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color (most common)
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    mid = cols // 2

    output = [row[:] for row in input_grid]

    for r in range(rows):
        left = input_grid[r][0]
        right = input_grid[r][cols - 1]

        if left == bg and right == bg:
            continue

        for c in range(cols):
            if c < mid:
                output[r][c] = left
            elif c == mid:
                output[r][c] = 4
            else:
                output[r][c] = right

    return output
