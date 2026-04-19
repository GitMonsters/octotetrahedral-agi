"""
Solver for ARC-AGI task 4f2f1437.

Rule:
1. Background = most frequent color.
2. Cluster color = least frequent color overall.
3. Cluster cells of that color using Chebyshev distance <= 2 (transitive).
4. For each cluster, compute bounding box.
5. Fill background cells within each bounding box with color 3.
"""

from collections import Counter, deque


def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])

    # Count colors
    color_counts: Counter = Counter()
    for r in range(H):
        for c in range(W):
            color_counts[grid[r][c]] += 1

    bg = color_counts.most_common(1)[0][0]
    cluster_color = color_counts.most_common()[-1][0]

    # Collect cluster color cells
    cells: set[tuple[int, int]] = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] == cluster_color:
                cells.add((r, c))

    # Cluster using Chebyshev distance <= 2
    visited: set[tuple[int, int]] = set()
    components: list[set[tuple[int, int]]] = []
    for cell in cells:
        if cell in visited:
            continue
        comp: set[tuple[int, int]] = set()
        queue = deque([cell])
        while queue:
            cr, cc = queue.popleft()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            comp.add((cr, cc))
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if (nr, nc) in cells and (nr, nc) not in visited:
                            queue.append((nr, nc))
        components.append(comp)

    # Build output: fill bg cells in each cluster's bbox with 3
    output = [row[:] for row in grid]
    for comp in components:
        rows = [r for r, c in comp]
        cols = [c for r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if output[r][c] == bg:
                    output[r][c] = 3

    return output
