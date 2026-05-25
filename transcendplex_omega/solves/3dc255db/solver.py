from collections import Counter, deque
from typing import List, Tuple

Grid = List[List[int]]
Point = Tuple[int, int]


def solve(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * width for _ in range(height)]
    components = []

    for row in range(height):
        for col in range(width):
            if output[row][col] == 0 or visited[row][col]:
                continue

            queue = deque([(row, col)])
            visited[row][col] = True
            component = []

            while queue:
                current_row, current_col = queue.popleft()
                component.append((current_row, current_col, output[current_row][current_col]))
                for delta_row in (-1, 0, 1):
                    for delta_col in (-1, 0, 1):
                        if delta_row == 0 and delta_col == 0:
                            continue
                        next_row = current_row + delta_row
                        next_col = current_col + delta_col
                        if not (0 <= next_row < height and 0 <= next_col < width):
                            continue
                        if visited[next_row][next_col] or output[next_row][next_col] == 0:
                            continue
                        visited[next_row][next_col] = True
                        queue.append((next_row, next_col))

            components.append(component)

    direction_map = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    for component in components:
        color_counts = Counter(value for _, _, value in component)
        if len(color_counts) < 2:
            continue

        scaffold_color = color_counts.most_common()[0][0]
        marker_color = color_counts.most_common()[-1][0]
        scaffold_cells = [(row, col) for row, col, value in component if value == scaffold_color]
        marker_cells = [(row, col) for row, col, value in component if value == marker_color]
        marker_count = len(marker_cells)
        marker_row = sum(row for row, _ in marker_cells) / marker_count
        marker_col = sum(col for _, col in marker_cells) / marker_count

        scaffold_rows = [row for row, _ in scaffold_cells]
        scaffold_cols = [col for _, col in scaffold_cells]
        min_row, max_row = min(scaffold_rows), max(scaffold_rows)
        min_col, max_col = min(scaffold_cols), max(scaffold_cols)

        candidates = [
            ("up", [(row, col) for row, col in scaffold_cells if row == min_row]),
            ("down", [(row, col) for row, col in scaffold_cells if row == max_row]),
            ("left", [(row, col) for row, col in scaffold_cells if col == min_col]),
            ("right", [(row, col) for row, col in scaffold_cells if col == max_col]),
        ]
        adjacent_width = {
            "up": sum(1 for row, _ in scaffold_cells if row == min_row + 1),
            "down": sum(1 for row, _ in scaffold_cells if row == max_row - 1),
            "left": sum(1 for _, col in scaffold_cells if col == min_col + 1),
            "right": sum(1 for _, col in scaffold_cells if col == max_col - 1),
        }

        def score(candidate: Tuple[str, List[Point]]) -> Tuple[int, int, int, float]:
            direction, edge_cells = candidate
            tip = max(edge_cells, key=lambda cell: (cell[0] - marker_row) ** 2 + (cell[1] - marker_col) ** 2)
            delta_row, delta_col = direction_map[direction]
            beam_row = tip[0] + delta_row
            beam_col = tip[1] + delta_col
            available = 0
            while 0 <= beam_row < height and 0 <= beam_col < width and output[beam_row][beam_col] == 0:
                available += 1
                beam_row += delta_row
                beam_col += delta_col
            farthest = max(((row - marker_row) ** 2 + (col - marker_col) ** 2) ** 0.5 for row, col in edge_cells)
            return (len(edge_cells), -adjacent_width[direction], -min(marker_count, available), -farthest)

        best_direction, best_edge = min(candidates, key=score)
        tip = max(best_edge, key=lambda cell: (cell[0] - marker_row) ** 2 + (cell[1] - marker_col) ** 2)
        delta_row, delta_col = direction_map[best_direction]

        for row, col in marker_cells:
            output[row][col] = 0

        beam_row = tip[0] + delta_row
        beam_col = tip[1] + delta_col
        placed = 0
        while 0 <= beam_row < height and 0 <= beam_col < width and placed < marker_count:
            if output[beam_row][beam_col] != 0:
                break
            output[beam_row][beam_col] = marker_color
            placed += 1
            beam_row += delta_row
            beam_col += delta_col

    return output
