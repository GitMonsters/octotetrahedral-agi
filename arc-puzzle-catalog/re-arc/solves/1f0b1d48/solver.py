from collections import Counter

def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    total = rows * cols

    color_counts = Counter()
    for row in input_grid:
        for cell in row:
            color_counts[cell] += 1

    bg_color, bg_count = color_counts.most_common(1)[0]
    bg_fraction = bg_count / total

    output = [[6] * cols for _ in range(rows)]

    if len(color_counts) == 1:
        # Single color → vertical line at column 0
        for i in range(rows):
            output[i][0] = 2
    elif bg_fraction > 0.5:
        # Background is strict majority → anti-diagonal
        for i in range(rows):
            output[i][cols - 1 - i] = 2
    else:
        # Background is not strict majority → main diagonal
        for i in range(rows):
            output[i][i] = 2

    return output
