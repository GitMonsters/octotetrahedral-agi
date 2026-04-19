import json
from collections import Counter

# Detect fill color from training data (appears in output but not input)
with open('/tmp/rearc45/19d15a12.json') as _f:
    _task = json.load(_f)

def _detect_fill_color(task):
    for pair in task['train']:
        inp_colors = {c for row in pair['input'] for c in row}
        out_colors = {c for row in pair['output'] for c in row}
        new = out_colors - inp_colors
        if new:
            return new.pop()
    return 4

FILL = _detect_fill_color(_task)


def transform(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])

    # Background = most common color
    counts: Counter = Counter()
    for row in grid:
        counts.update(row)
    bg = counts.most_common(1)[0][0]

    output = [row[:] for row in grid]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                color = grid[r][c]
                # Draw alternating bar from col 0 to col c
                for j in range(c + 1):
                    if (c - j) % 2 == 0:
                        output[r][j] = color
                    else:
                        output[r][j] = FILL
                break  # one non-bg pixel per row

    return output
