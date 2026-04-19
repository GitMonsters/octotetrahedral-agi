from collections import Counter, defaultdict
from typing import DefaultDict, List, Tuple


Grid = List[List[int]]
Bar = Tuple[int, int, int]


def find_bars(grid: Grid, background: int) -> List[Bar]:
    runs: List[Tuple[int, int, int, int]] = []
    width = len(grid[0])
    for row_index, row in enumerate(grid):
        col = 0
        while col < width:
            color = row[col]
            end = col
            while end < width and row[end] == color:
                end += 1
            if color != background and end - col >= 3:
                runs.append((end - col, row_index, col, end - 1))
            col = end
    runs.sort(reverse=True)

    bars: List[Bar] = []
    used_rows = set()
    for _, row_index, start, end in runs:
        if row_index not in used_rows:
            bars.append((row_index, start, end))
            used_rows.add(row_index)
        if len(bars) == 2:
            break
    bars.sort()
    return bars


def choose_bar(bars: List[Bar], row: int, col: int) -> int:
    candidates = [
        index
        for index, (_, start, end) in enumerate(bars)
        if start <= col <= end
    ]
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=lambda index: abs(row - bars[index][0]))


def fill_vertical(grid: Grid, col: int, row_a: int, row_b: int, color: int) -> None:
    step = 1 if row_b > row_a else -1
    for row in range(row_a + step, row_b, step):
        grid[row][col] = color


def project_to_other_bar(
    grid: Grid,
    source_bar: Bar,
    target_bar: Bar,
    relative_col: int,
    color: int,
) -> None:
    source_row, _, _ = source_bar
    target_row, target_start, target_end = target_bar
    target_col = target_start + relative_col
    if not (target_start <= target_col <= target_end):
        return
    if target_row < source_row:
        rows = range(target_row + 1, source_row + 1)
    else:
        rows = range(source_row, target_row)
    for row in rows:
        grid[row][target_col] = color


def transform(grid: Grid) -> Grid:
    height = len(grid)
    width = len(grid[0])
    background = Counter(value for row in grid for value in row).most_common(1)[0][0]
    bars = find_bars(grid, background)
    if len(bars) != 2:
        return [row[:] for row in grid]

    output = [row[:] for row in grid]
    bar_cells = {
        (row, col)
        for row, start, end in bars
        for col in range(start, end + 1)
    }
    projections: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)

    for row in range(height):
        for col in range(width):
            color = grid[row][col]
            if color == background or (row, col) in bar_cells:
                continue
            bar_index = choose_bar(bars, row, col)
            bar_row, bar_start, _ = bars[bar_index]
            output[row][col] = 5
            fill_vertical(output, col, row, bar_row, color)
            projections[bar_index].append((col - bar_start, color))

    for bar_index, items in projections.items():
        other_index = 1 - bar_index
        for relative_col, color in items:
            project_to_other_bar(output, bars[bar_index], bars[other_index], relative_col, color)

    return output
