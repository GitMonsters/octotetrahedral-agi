def transform(input_grid):
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color (most common)
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find all non-background dots
    dots = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                dots.append((r, c))

    # Create output grid (all background)
    output = [[bg] * cols for _ in range(rows)]

    # For each dot, place the 4 diagonal markers
    for r, c in dots:
        if r - 1 >= 0 and c - 1 >= 0:
            output[r - 1][c - 1] = 0  # black
        if r - 1 >= 0 and c + 1 < cols:
            output[r - 1][c + 1] = 2  # red
        if r + 1 < rows and c - 1 >= 0:
            output[r + 1][c - 1] = 7  # orange
        if r + 1 < rows and c + 1 < cols:
            output[r + 1][c + 1] = 1  # blue

    return output
