from collections import deque
from typing import Dict, List, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]
Component = Dict[str, object]


def _components(grid: Grid, color: int) -> List[Component]:
    height, width = len(grid), len(grid[0])
    seen = [[False] * width for _ in range(height)]
    components: List[Component] = []

    for row in range(height):
        for col in range(width):
            if grid[row][col] != color or seen[row][col]:
                continue

            queue = deque([(row, col)])
            seen[row][col] = True
            cells: List[Cell] = []

            while queue:
                cur_row, cur_col = queue.popleft()
                cells.append((cur_row, cur_col))
                for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt_row = cur_row + d_row
                    nxt_col = cur_col + d_col
                    if (
                        0 <= nxt_row < height
                        and 0 <= nxt_col < width
                        and not seen[nxt_row][nxt_col]
                        and grid[nxt_row][nxt_col] == color
                    ):
                        seen[nxt_row][nxt_col] = True
                        queue.append((nxt_row, nxt_col))

            min_row = min(r for r, _ in cells)
            min_col = min(c for _, c in cells)
            max_row = max(r for r, _ in cells)
            max_col = max(c for _, c in cells)
            shape = [(r - min_row, c - min_col) for r, c in cells]
            components.append(
                {
                    "min_col": min_col,
                    "height": max_row - min_row + 1,
                    "width": max_col - min_col + 1,
                    "shape": shape,
                }
            )

    return components


def _paint(grid: Grid, color: int, shape: List[Cell], base_row: int, base_col: int) -> None:
    for d_row, d_col in shape:
        grid[base_row + d_row][base_col + d_col] = color


def solve(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    colors = sorted({value for row in grid for value in row if value != 0})

    color_entries = []
    for color in colors:
        components = _components(grid, color)
        side_key = min(component["min_col"] for component in components)
        color_entries.append((side_key, color, components))

    color_entries.sort(key=lambda entry: entry[0])
    left_color, left_components = color_entries[0][1], color_entries[0][2]
    right_color, right_components = color_entries[1][1], color_entries[1][2]
    right_start_col = max(component["width"] for component in left_components) + 1

    output = [[0] * width for _ in range(height)]

    for start_col, color, components in (
        (0, left_color, left_components),
        (right_start_col, right_color, right_components),
    ):
        top_component, bottom_component = sorted(
            components,
            key=lambda component: component["min_col"],
            reverse=True,
        )
        top_row = height - (top_component["height"] + 1 + bottom_component["height"])
        bottom_row = top_row + top_component["height"] + 1
        _paint(output, color, top_component["shape"], top_row, start_col)
        _paint(output, color, bottom_component["shape"], bottom_row, start_col)

    return output
