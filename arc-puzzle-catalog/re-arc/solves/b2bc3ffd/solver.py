"""ARC-AGI puzzle b2bc3ffd solver.

Rule: Each colored object (non-7, non-8) sitting above the floor (row of 8s)
moves UP by the number of cells it contains.
"""

import copy
from collections import deque


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])
    bg = 7

    # Start with background everywhere, then restore floor row
    output = [[bg] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] == 8:
                output[r][c] = 8

    # Find connected components of non-background, non-floor cells via BFS
    visited = [[False] * cols for _ in range(rows)]
    objects: list[list[tuple[int, int, int]]] = []  # list of [(r, c, color), ...]

    for r in range(rows):
        for c in range(cols):
            val = input_grid[r][c]
            if val == bg or val == 8 or visited[r][c]:
                continue
            # BFS to find all cells of this object (same color, connected)
            component: list[tuple[int, int, int]] = []
            queue = deque([(r, c)])
            visited[r][c] = True
            color = val
            while queue:
                cr, cc = queue.popleft()
                component.append((cr, cc, input_grid[cr][cc]))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                        if input_grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
            objects.append(component)

    # Move each object up by its cell count
    for component in objects:
        shift = len(component)
        for r, c, color in component:
            new_r = r - shift
            if 0 <= new_r < rows:
                output[new_r][c] = color

    return output


# ── Testing ──────────────────────────────────────────────────────────────────

examples = [
    {
        "input": [
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [7,9,7,7,7,7,7,7],
            [9,9,9,7,7,2,2,2],
            [8,8,8,8,8,8,8,8],
        ],
        "output": [
            [7,7,7,7,7,7,7,7],
            [7,9,7,7,7,7,7,7],
            [9,9,9,7,7,7,7,7],
            [7,7,7,7,7,2,2,2],
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [8,8,8,8,8,8,8,8],
        ],
    },
    {
        "input": [
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [7,7,7,9,7,7,7,7],
            [7,2,7,9,7,7,7,3],
            [7,2,7,9,7,1,7,3],
            [8,8,8,8,8,8,8,8],
        ],
        "output": [
            [7,7,7,7,7,7,7,7],
            [7,7,7,9,7,7,7,7],
            [7,7,7,9,7,7,7,7],
            [7,2,7,9,7,7,7,3],
            [7,2,7,7,7,7,7,3],
            [7,7,7,7,7,1,7,7],
            [7,7,7,7,7,7,7,7],
            [8,8,8,8,8,8,8,8],
        ],
    },
    {
        "input": [
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,4],
            [1,1,7,7,7,3,7,4],
            [1,1,1,7,3,3,7,4],
            [8,8,8,8,8,8,8,8],
        ],
        "output": [
            [1,1,7,7,7,7,7,7],
            [1,1,1,7,7,7,7,4],
            [7,7,7,7,7,3,7,4],
            [7,7,7,7,3,3,7,4],
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [7,7,7,7,7,7,7,7],
            [8,8,8,8,8,8,8,8],
        ],
    },
]

test_case = {
    "input": [
        [7,7,7,7,7,7,7,7],
        [7,7,7,7,7,7,7,7],
        [7,7,7,7,7,7,7,7],
        [7,7,7,7,7,7,7,7],
        [7,7,7,7,7,7,7,7],
        [7,7,7,7,7,7,7,7],
        [5,5,5,5,5,5,7,6],
        [8,8,8,8,8,8,8,8],
    ],
    "output": [
        [5,5,5,5,5,5,7,7],
        [7,7,7,7,7,7,7,7],
        [7,7,7,7,7,7,7,7],
        [7,7,7,7,7,7,7,7],
        [7,7,7,7,7,7,7,7],
        [7,7,7,7,7,7,7,6],
        [7,7,7,7,7,7,7,7],
        [8,8,8,8,8,8,8,8],
    ],
}

all_pass = True
for i, ex in enumerate(examples + [test_case]):
    result = transform(ex["input"])
    if result == ex["output"]:
        print(f"{'Example' if i < 3 else 'Test'} {i}: PASS")
    else:
        all_pass = False
        print(f"{'Example' if i < 3 else 'Test'} {i}: FAIL")
        for r, (got, exp) in enumerate(zip(result, ex["output"])):
            if got != exp:
                print(f"  Row {r}: got {got} expected {exp}")

print("\nSOLVED" if all_pass else "\nFAILED")
