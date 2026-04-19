def transform(input_grid: list[list[int]]) -> list[list[int]]:
    from collections import Counter, deque

    grid = input_grid
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]

    # Find connected components of non-bg cells
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and (r, c) not in visited:
                comp = set()
                q = deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    comp.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] != bg:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                components.append(comp)

    output = [[bg] * cols for _ in range(rows)]

    for comp in components:
        # Find all 2x2 block anchor positions (top-left corners)
        anchors = []
        for r, c in sorted(comp):
            if (r + 1, c) in comp and (r, c + 1) in comp and (r + 1, c + 1) in comp:
                anchors.append((r, c))

        # Find maximum non-overlapping 2x2 tiling via backtracking
        # Collect ALL tilings with maximum coverage, then pick best
        max_cov = [0]
        all_best = []

        def backtrack(idx: int, used: set, covered: set) -> None:
            if len(covered) > max_cov[0]:
                max_cov[0] = len(covered)
                all_best.clear()
                all_best.append(frozenset(covered))
            elif len(covered) == max_cov[0] and len(covered) > 0:
                all_best.append(frozenset(covered))
            for i in range(idx, len(anchors)):
                r, c = anchors[i]
                cells = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
                if not (cells & used):
                    backtrack(i + 1, used | cells, covered | cells)

        backtrack(0, set(), set())

        # Among tilings with max coverage, pick the one where remaining
        # cells are best connected (fewest connected components)
        def count_remaining_components(covered: frozenset) -> int:
            remaining = comp - covered
            if not remaining:
                return 0
            seen = set()
            num = 0
            for start in remaining:
                if start in seen:
                    continue
                num += 1
                stack = [start]
                seen.add(start)
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (nr, nc) in remaining and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
            return num

        best_cover = set()
        if all_best:
            # Pick tiling with fewest remaining connected components
            best_cover = min(all_best, key=count_remaining_components)

        # Color: covered by 2x2 blocks → 8, rest → 4
        for r, c in comp:
            output[r][c] = 8 if (r, c) in best_cover else 4

    return output
