from collections import Counter
from typing import Dict, List, Tuple

Grid = List[List[int]]
Point = Tuple[int, int]


def _non_background_components(grid: Grid, background: int) -> List[Dict[str, object]]:
    height, width = len(grid), len(grid[0])
    visited: set[Point] = set()
    components: List[Dict[str, object]] = []

    for row in range(height):
        for col in range(width):
            if (row, col) in visited or grid[row][col] == background:
                continue

            stack = [(row, col)]
            visited.add((row, col))
            cells: List[Point] = []
            colors: Counter[int] = Counter()

            while stack:
                current_row, current_col = stack.pop()
                cells.append((current_row, current_col))
                colors[grid[current_row][current_col]] += 1

                for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in visited or grid[next_row][next_col] == background:
                        continue
                    visited.add((next_row, next_col))
                    stack.append((next_row, next_col))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            components.append(
                {
                    "cells": cells,
                    "size": len(cells),
                    "center": (sum(rows) / len(cells), sum(cols) / len(cells)),
                    "colors": colors,
                }
            )

    return components


def _cluster_into_two_groups(components: List[Dict[str, object]]) -> List[List[Dict[str, object]]]:
    if len(components) <= 1:
        return [components, []]

    best_distance = -1.0
    start_pair = (0, 1)
    for first in range(len(components)):
        first_center = components[first]["center"]
        for second in range(first + 1, len(components)):
            second_center = components[second]["center"]
            distance = (first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2
            if distance > best_distance:
                best_distance = distance
                start_pair = (first, second)

    centers = [list(components[start_pair[0]]["center"]), list(components[start_pair[1]]["center"])]

    for _ in range(30):
        groups: List[List[Dict[str, object]]] = [[], []]
        for component in components:
            distances = [
                (component["center"][0] - center[0]) ** 2 + (component["center"][1] - center[1]) ** 2
                for center in centers
            ]
            groups[0 if distances[0] <= distances[1] else 1].append(component)

        next_centers: List[List[float]] = []
        for index, group in enumerate(groups):
            if not group:
                next_centers.append(centers[index])
                continue
            total_size = sum(component["size"] for component in group)
            next_centers.append(
                [
                    sum(component["center"][0] * component["size"] for component in group) / total_size,
                    sum(component["center"][1] * component["size"] for component in group) / total_size,
                ]
            )

        if max(abs(next_centers[index][axis] - centers[index][axis]) for index in (0, 1) for axis in (0, 1)) < 1e-9:
            return groups
        centers = next_centers

    return groups


def _masked_crop(grid: Grid, group: List[Dict[str, object]], background: int) -> Grid:
    cells = {cell for component in group for cell in component["cells"]}
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)

    crop: Grid = []
    for row in range(top, bottom + 1):
        crop_row: List[int] = []
        for col in range(left, right + 1):
            crop_row.append(grid[row][col] if (row, col) in cells else background)
        crop.append(crop_row)
    return crop


def _regions_without_scaffold(grid: Grid, scaffold: int) -> List[Dict[str, object]]:
    height, width = len(grid), len(grid[0])
    visited: set[Point] = set()
    regions: List[Dict[str, object]] = []

    for row in range(height):
        for col in range(width):
            if (row, col) in visited or grid[row][col] == scaffold:
                continue

            stack = [(row, col)]
            visited.add((row, col))
            cells: List[Point] = []
            colors: Counter[int] = Counter()

            while stack:
                current_row, current_col = stack.pop()
                cells.append((current_row, current_col))
                colors[grid[current_row][current_col]] += 1

                for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in visited or grid[next_row][next_col] == scaffold:
                        continue
                    visited.add((next_row, next_col))
                    stack.append((next_row, next_col))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            regions.append(
                {
                    "cells": cells,
                    "center": (sum(rows) / len(cells), sum(cols) / len(cells)),
                    "colors": colors,
                }
            )

    return regions


def solve(grid: Grid) -> Grid:
    flat = [value for row in grid for value in row]
    background = Counter(flat).most_common(1)[0][0]
    scaffold = Counter(value for value in flat if value != background).most_common(1)[0][0]

    components = _non_background_components(grid, background)
    groups = _cluster_into_two_groups(components)
    if not groups[1]:
        return [row[:] for row in grid]

    group_palettes: List[set[int]] = []
    for group in groups:
        palette: Counter[int] = Counter()
        for component in group:
            palette += component["colors"]
        group_palettes.append(set(palette))

    source_index = 0 if len(group_palettes[0] - {background, scaffold}) >= len(group_palettes[1] - {background, scaffold}) else 1
    target_index = 1 - source_index

    source = _masked_crop(grid, groups[source_index], background)
    target = _masked_crop(grid, groups[target_index], background)
    source_regions = _regions_without_scaffold(source, scaffold)
    target_regions = _regions_without_scaffold(target, scaffold)

    source_height, source_width = len(source), len(source[0])
    target_height, target_width = len(target), len(target[0])
    output = [row[:] for row in target]

    decorated_regions: List[Tuple[Dict[str, object], int]] = []
    for region in source_regions:
        decoration_counts = {color: count for color, count in region["colors"].items() if color != background}
        if not decoration_counts:
            continue
        color = max(decoration_counts, key=decoration_counts.get)
        decorated_regions.append((region, color))

    unused_targets = set(range(len(target_regions)))
    decorated_regions.sort(key=lambda item: (item[0]["center"][0] / source_height, item[0]["center"][1] / source_width))

    for region, color in decorated_regions:
        source_row = region["center"][0] / source_height
        source_col = region["center"][1] / source_width
        best_target = min(
            unused_targets,
            key=lambda index: (
                (target_regions[index]["center"][0] / target_height - source_row) ** 2
                + (target_regions[index]["center"][1] / target_width - source_col) ** 2
            ),
        )
        unused_targets.remove(best_target)
        for row, col in target_regions[best_target]["cells"]:
            output[row][col] = color

    return output
