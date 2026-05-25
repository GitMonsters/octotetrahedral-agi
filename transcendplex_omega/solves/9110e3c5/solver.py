from collections import Counter


PATTERNS = {
    1: [[0, 0, 8], [8, 8, 0], [0, 8, 0]],
    2: [[0, 0, 0], [8, 8, 8], [0, 0, 0]],
    3: [[0, 8, 8], [0, 8, 0], [0, 8, 0]],
}


def solve(grid: list[list[int]]) -> list[list[int]]:
    counts = Counter(v for row in grid for v in row if v != 0)
    dominant = counts.most_common(1)[0][0]
    return PATTERNS[dominant]
