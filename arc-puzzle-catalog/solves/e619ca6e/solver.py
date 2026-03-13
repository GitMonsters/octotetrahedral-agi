def solve(grid: list[list[int]]) -> list[list[int]]:
    """Each rectangular block of 3s spawns an expanding V-pattern: at each level n,
    two copies of the block are placed n*W left and n*W right, shifted n*H rows down,
    clipped to grid boundaries."""
    rows, cols = len(grid), len(grid[0])

    # Find rectangular blocks of 3s via flood-fill
    visited: set[tuple[int, int]] = set()
    blocks: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3 and (r, c) not in visited:
                min_r, max_r, min_c, max_c = r, r, c, c
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == 3:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                            min_r, max_r = min(min_r, nr), max(max_r, nr)
                            min_c, max_c = min(min_c, nc), max(max_c, nc)
                blocks.append((min_r, min_c, max_c - min_c + 1, max_r - min_r + 1))

    result = [[0] * cols for _ in range(rows)]

    for br, bc, w, h in blocks:
        # Draw original block + expanding V-arms
        for n in range(200):
            top = br + n * h
            if top >= rows:
                break
            if n == 0:
                arms = [bc]
            else:
                arms = [bc - n * w, bc + n * w]
            for arm_c in arms:
                if arm_c + w <= 0 or arm_c >= cols:
                    continue
                for rr in range(top, min(top + h, rows)):
                    for cc in range(max(0, arm_c), min(cols, arm_c + w)):
                        result[rr][cc] = 3

    return result
