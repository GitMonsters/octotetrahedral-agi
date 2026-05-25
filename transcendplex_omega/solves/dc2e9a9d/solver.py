from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Each shape is a rectangular frame (color 3) with a single-cell bump.
    For each shape, create a mirror copy on the opposite side of the bump:
    - Horizontal bumps (left/right) → mirror colored 1
    - Vertical bumps (top/bottom) → mirror colored 8
    The mirror is reflected across a line 1 cell beyond the body edge opposite the bump.
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find connected components of 3s
    visited = [[False] * cols for _ in range(rows)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3 and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 3:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(comp)

    for comp in components:
        comp_set = set(comp)

        # Find bump cell (degree 1 in component graph)
        bump = None
        for r, c in comp:
            neighbors = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (r + dr, c + dc) in comp_set)
            if neighbors == 1:
                bump = (r, c)
                break

        if bump is None:
            continue

        # Body = component minus bump
        body = [cell for cell in comp if cell != bump]
        body_min_r = min(r for r, c in body)
        body_max_r = max(r for r, c in body)
        body_min_c = min(c for r, c in body)
        body_max_c = max(c for r, c in body)

        br, bc = bump

        # Determine bump direction
        if br < body_min_r:
            direction = "top"
        elif br > body_max_r:
            direction = "bottom"
        elif bc < body_min_c:
            direction = "left"
        else:
            direction = "right"

        # Horizontal bumps → color 1, vertical bumps → color 8
        color = 1 if direction in ("left", "right") else 8

        # Reflect across axis 1 cell beyond the body edge opposite the bump
        if direction == "left":
            center = body_max_c + 1
            for r, c in comp:
                nc = 2 * center - c
                if 0 <= nc < cols:
                    result[r][nc] = color
        elif direction == "right":
            center = body_min_c - 1
            for r, c in comp:
                nc = 2 * center - c
                if 0 <= nc < cols:
                    result[r][nc] = color
        elif direction == "top":
            center = body_max_r + 1
            for r, c in comp:
                nr = 2 * center - r
                if 0 <= nr < rows:
                    result[nr][c] = color
        elif direction == "bottom":
            center = body_min_r - 1
            for r, c in comp:
                nr = 2 * center - r
                if 0 <= nr < rows:
                    result[nr][c] = color

    return result
