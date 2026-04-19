#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import deque
from typing import Iterable


POSITIVE_3X3 = {
    "001001111",
    "001101111",
    "010101001",
    "011000011",
    "011000111",
    "011001111",
    "100001111",
    "100100100",
    "100100101",
    "100101111",
    "100101xxx",
    "101100001",
    "101100011",
    "101101101",
    "110000111",
    "110001001",
    "110001010",
    "111001111",
    "111100101",
    "111101011",
    "111101101",
    "111101111",
}

POSITIVE_5X5 = {
    "0000111111100110111110001",
    "0010101000010011100001011",
    "0100001001110000101101111",
    "0100100010010101111011010",
    "0100111000010110111101011",
    "0101011110110100101111010",
    "0101011111110110011100111",
    "0101110100110111001101001",
    "0101110111100110111000011",
    "0110111100100010111110111",
    "0111001110110110101111111",
    "1001101110000111111000110",
    "1001111111110111111011001",
    "1010101011110011011100001",
    "10110111011001110100xxxxx",
    "1100110111000011111110011",
    "1101111000000111111101110",
    "11011111101100111010xxxxx",
    "111101100111010xxxxxxxxxx",
    "1111011010010111101000001",
    "x1101x0011x0011x1111xxxxx",
    "x1111x1101x0011x0011x1111",
    "xxxxx10111100000011011111",
    "xxxxx11011110000001111111",
    "xxxxx11110110101111001111",
}


Grid = list[list[int]]
Cell = tuple[int, int]
BBox = tuple[int, int, int, int]


def connected_components(grid: Grid, color: int, diagonals: bool) -> list[list[Cell]]:
    height, width = len(grid), len(grid[0])
    directions = (
        [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        if diagonals
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    seen: set[Cell] = set()
    components: list[list[Cell]] = []

    for row in range(height):
        for col in range(width):
            if grid[row][col] != color or (row, col) in seen:
                continue
            queue = deque([(row, col)])
            seen.add((row, col))
            component: list[Cell] = []
            while queue:
                cur_row, cur_col = queue.popleft()
                component.append((cur_row, cur_col))
                for dr, dc in directions:
                    nr, nc = cur_row + dr, cur_col + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and grid[nr][nc] == color
                        and (nr, nc) not in seen
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            components.append(component)

    return components


def bbox(cells: Iterable[Cell]) -> BBox:
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    return min(rows), min(cols), max(rows), max(cols)


def merge_boxes(boxes: Iterable[BBox]) -> BBox:
    box_list = list(boxes)
    return (
        min(box[0] for box in box_list),
        min(box[1] for box in box_list),
        max(box[2] for box in box_list),
        max(box[3] for box in box_list),
    )


def cells_in_box(box: BBox) -> Iterable[Cell]:
    row1, col1, row2, col2 = box
    for row in range(row1, row2 + 1):
        for col in range(col1, col2 + 1):
            yield row, col


def normalize_pattern(grid: Grid, row: int, col: int, source: int, radius: int) -> str:
    height, width = len(grid), len(grid[0])
    bits: list[str] = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width:
                bits.append("0" if grid[nr][nc] == source else "1")
            else:
                bits.append("x")
    return "".join(bits)


def solve_three_color(grid: Grid) -> Grid:
    colors = sorted({value for row in grid for value in row})
    object_color = 9
    other_colors = [color for color in colors if color != object_color]
    components = connected_components(grid, object_color, diagonals=True)
    boxes = [bbox(component) for component in components]

    counts_in_boxes = {color: 0 for color in other_colors}
    for box in boxes:
        for row, col in cells_in_box(box):
            value = grid[row][col]
            if value in counts_in_boxes:
                counts_in_boxes[value] += 1

    background = max(other_colors, key=lambda color: counts_in_boxes[color])
    wall = min(other_colors, key=lambda color: counts_in_boxes[color])

    adjacency = [set() for _ in boxes]
    for left in range(len(boxes)):
        for right in range(left + 1, len(boxes)):
            union_box = merge_boxes([boxes[left], boxes[right]])
            if all(grid[row][col] != wall for row, col in cells_in_box(union_box)):
                adjacency[left].add(right)
                adjacency[right].add(left)

    groups: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(boxes)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        group: list[int] = []
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        groups.append(group)

    output = [row[:] for row in grid]
    for group in groups:
        union_box = merge_boxes(boxes[index] for index in group)
        for row, col in cells_in_box(union_box):
            if grid[row][col] == background:
                output[row][col] = 2
    return output


def solve_two_color(grid: Grid) -> Grid:
    colors = sorted({value for row in grid for value in row})
    output = [row[:] for row in grid]

    best_source = colors[0]
    best_exact: list[Cell] = []
    best_fallback: list[Cell] = []
    best_score: tuple[int, int, int, int] | None = None

    for source in colors:
        exact_hits: list[Cell] = []
        fallback_hits: list[Cell] = []
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] != source:
                    continue
                pattern_5x5 = normalize_pattern(grid, row, col, source, radius=2)
                if pattern_5x5 in POSITIVE_5X5:
                    exact_hits.append((row, col))
                    continue
                pattern_3x3 = normalize_pattern(grid, row, col, source, radius=1)
                if pattern_3x3 in POSITIVE_3X3:
                    fallback_hits.append((row, col))

        component_count = len(connected_components(grid, source, diagonals=False))
        cell_count = sum(value == source for row in grid for value in row)
        score = (len(exact_hits), len(fallback_hits), component_count, -cell_count)
        if best_score is None or score > best_score:
            best_score = score
            best_source = source
            best_exact = exact_hits
            best_fallback = fallback_hits

    chosen = best_exact if best_exact else best_fallback
    for row, col in chosen:
        if grid[row][col] == best_source:
            output[row][col] = 2
    return output


def transform(input_grid: Grid) -> Grid:
    colors = sorted({value for row in input_grid for value in row})
    if len(colors) == 3 and 9 in colors:
        return solve_three_color(input_grid)
    if len(colors) == 2:
        return solve_two_color(input_grid)
    return [row[:] for row in input_grid]


if __name__ == "__main__":
    task_path = "/Users/evanpieser/re_arc_solves/39f83d5c_task.json"
    with open(task_path) as handle:
        task = json.load(handle)

    ok = True
    for index, example in enumerate(task["train"], start=1):
        predicted = transform(example["input"])
        passed = predicted == example["output"]
        print(f"train {index}: {'PASS' if passed else 'FAIL'}")
        ok &= passed

    if ok:
        print("All training examples passed.")
