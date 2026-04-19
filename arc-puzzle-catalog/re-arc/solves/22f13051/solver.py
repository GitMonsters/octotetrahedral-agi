def transform(input_grid):
    from collections import Counter

    N = len(input_grid)

    # Find background color (most common)
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find bounding box of non-background cells
    non_bg = [(r, c) for r in range(N) for c in range(N) if input_grid[r][c] != bg]

    min_r = min(r for r, c in non_bg)
    max_r = max(r for r, c in non_bg)
    min_c = min(c for r, c in non_bg)
    max_c = max(c for r, c in non_bg)

    # Determine flips to move pattern to bottom-right corner
    flip_v = min_r == 0 and max_r != N - 1
    flip_h = min_c == 0 and max_c != N - 1

    grid = [row[:] for row in input_grid]
    if flip_v:
        grid = grid[::-1]
    if flip_h:
        grid = [row[::-1] for row in grid]

    # Read main diagonal and extract repeating sequence
    diag = [grid[i][i] for i in range(N)]
    period = N // 2
    seq = diag[period:]

    # Generate 2N x 2N output
    out_size = N * 2
    output = [[0] * out_size for _ in range(out_size)]
    for r in range(out_size):
        for c in range(out_size):
            output[r][c] = seq[min(r, c) % period]

    # Reverse flips
    if flip_h:
        output = [row[::-1] for row in output]
    if flip_v:
        output = output[::-1]

    return output
