from collections import Counter, defaultdict
from typing import DefaultDict, List, Tuple

Grid = List[List[int]]
Signature = Tuple[int, int, int, int]


def find_row_signatures(grid: Grid) -> DefaultDict[Signature, List[int]]:
    signatures: DefaultDict[Signature, List[int]] = defaultdict(list)
    width = len(grid[0])
    for row_index, row in enumerate(grid):
        for left in range(width - 2):
            edge = row[left]
            fill = row[left + 1]
            if edge == fill:
                continue
            right = left + 1
            while right < width and row[right] == fill:
                right += 1
            if right < width and row[right] == edge and right - left >= 2:
                signatures[(edge, fill, left, right)].append(row_index)
    return signatures


def choose_signature(grid: Grid) -> Signature | None:
    counts = Counter(value for row in grid for value in row)
    best_signature: Signature | None = None
    best_score: Tuple[int, int, int, int] | None = None

    for signature, rows in find_row_signatures(grid).items():
        if len(rows) < 2:
            continue
        edge, fill, left, right = signature
        height = max(rows) - min(rows) + 1
        width = right - left + 1
        area = height * width
        rarity = -min(counts[edge], counts[fill])
        score = (rarity, area, len(rows), width)
        if best_score is None or score > best_score:
            best_score = score
            best_signature = signature
    return best_signature


def transform(grid: Grid) -> Grid:
    signature = choose_signature(grid)
    if signature is None:
        return [row[:] for row in grid]

    rows = find_row_signatures(grid)[signature]
    _, _, left, right = signature
    top = min(rows)
    bottom = max(rows)
    return [row[left : right + 1] for row in grid[top : bottom + 1]]
