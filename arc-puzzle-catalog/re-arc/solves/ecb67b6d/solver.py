"""ARC puzzle ecb67b6d solver.

Rule: Find 8-connected components of gray(5) cells. If a component contains
3+ consecutive cells on any diagonal line, recolor the entire component to cyan(8).
"""
from collections import deque


def find_8conn_components(grid, val):
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    comps = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == val and not visited[r][c]:
                comp = set()
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.add((cr, cc))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == val:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                comps.append(comp)
    return comps


def has_diagonal_run_3(comp):
    for r, c in comp:
        for dr, dc in [(1, 1), (1, -1)]:
            if (r + dr, c + dc) in comp and (r + 2 * dr, c + 2 * dc) in comp:
                return True
    return False


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    output = [row[:] for row in input_grid]
    comps = find_8conn_components(input_grid, 5)
    for comp in comps:
        if has_diagonal_run_3(comp):
            for r, c in comp:
                output[r][c] = 8
    return output


if __name__ == "__main__":
    test_input = [
        [7,5,5,7,7,7,7,7,5,5,5,5,7,7,5,7,7,7,7],
        [7,7,5,7,7,7,7,7,7,7,7,5,5,5,7,7,5,7,7],
        [7,7,5,7,7,5,7,5,7,5,7,7,7,7,7,5,5,7,7],
        [5,7,7,5,7,7,7,5,5,7,7,7,7,5,7,5,7,5,7],
        [7,7,7,5,7,5,7,7,7,5,7,7,7,7,7,7,7,5,5],
        [7,5,7,5,7,7,7,7,7,7,5,7,7,7,7,7,7,5,5],
        [7,5,5,7,5,7,7,7,7,7,7,7,7,7,5,5,7,7,5],
        [7,7,7,7,7,7,5,5,7,7,7,5,5,7,7,7,7,5,7],
        [7,7,5,7,7,5,5,7,5,5,7,7,5,5,7,7,7,7,7],
        [7,7,7,7,7,7,5,5,7,7,5,7,7,7,7,7,5,7,7],
        [5,7,7,7,7,7,7,7,7,7,5,7,5,7,7,7,7,5,5],
        [7,5,5,5,7,7,7,7,5,7,7,7,7,7,7,7,7,5,7],
        [7,5,7,7,7,7,7,7,5,5,7,7,7,7,5,7,7,5,5],
        [7,5,5,7,7,7,5,7,5,5,7,5,5,5,7,5,7,7,7],
        [7,7,7,7,7,5,5,7,5,7,7,7,7,7,5,5,5,7,5],
        [7,7,7,7,7,7,5,5,7,7,7,7,7,7,5,7,7,7,7],
    ]

    expected = [
        [7,5,5,7,7,7,7,7,5,5,5,5,7,7,5,7,7,7,7],
        [7,7,5,7,7,7,7,7,7,7,7,5,5,5,7,7,8,7,7],
        [7,7,5,7,7,5,7,8,7,8,7,7,7,7,7,8,8,7,7],
        [5,7,7,5,7,7,7,8,8,7,7,7,7,5,7,8,7,8,7],
        [7,7,7,5,7,5,7,7,7,8,7,7,7,7,7,7,7,8,8],
        [7,5,7,5,7,7,7,7,7,7,8,7,7,7,7,7,7,8,8],
        [7,5,5,7,5,7,7,7,7,7,7,7,7,7,5,5,7,7,8],
        [7,7,7,7,7,7,5,5,7,7,7,5,5,7,7,7,7,8,7],
        [7,7,5,7,7,5,5,7,5,5,7,7,5,5,7,7,7,7,7],
        [7,7,7,7,7,7,5,5,7,7,5,7,7,7,7,7,5,7,7],
        [5,7,7,7,7,7,7,7,7,7,5,7,5,7,7,7,7,5,5],
        [7,5,5,5,7,7,7,7,8,7,7,7,7,7,7,7,7,5,7],
        [7,5,7,7,7,7,7,7,8,8,7,7,7,7,8,7,7,5,5],
        [7,5,5,7,7,7,8,7,8,8,7,8,8,8,7,8,7,7,7],
        [7,7,7,7,7,8,8,7,8,7,7,7,7,7,8,8,8,7,5],
        [7,7,7,7,7,7,8,8,7,7,7,7,7,7,8,7,7,7,7],
    ]

    result = transform(test_input)
    if result == expected:
        print("SOLVED")
    else:
        print("FAILED")
