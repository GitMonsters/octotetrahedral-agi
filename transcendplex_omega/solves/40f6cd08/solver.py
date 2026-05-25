def solve(grid: list[list[int]]) -> list[list[int]]:
    """Fill uniform rectangles with the concentric layer pattern from the template rectangle."""
    from collections import deque
    import copy

    result = copy.deepcopy(grid)
    H, W = len(grid), len(grid[0])

    # Find connected non-zero regions via BFS
    visited = [[False] * W for _ in range(H)]
    regions = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0 and not visited[r][c]:
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                rmin = min(r for r, c in cells)
                rmax = max(r for r, c in cells)
                cmin = min(c for r, c in cells)
                cmax = max(c for r, c in cells)
                colors = set(grid[r][c] for r, c in cells)
                regions.append({
                    'rmin': rmin, 'rmax': rmax, 'cmin': cmin, 'cmax': cmax,
                    'h': rmax - rmin + 1, 'w': cmax - cmin + 1,
                    'colors': colors, 'uniform': len(colors) == 1,
                })

    # Template = multi-colored region; targets = uniform regions
    template = None
    targets = []
    for reg in regions:
        if not reg['uniform']:
            template = reg
        else:
            targets.append(reg)

    if template is None:
        return result

    # Extract template subgrid
    sub = []
    for r in range(template['rmin'], template['rmax'] + 1):
        row = []
        for c in range(template['cmin'], template['cmax'] + 1):
            row.append(grid[r][c])
        sub.append(row)

    # Peel concentric layers from the template
    layers = []  # list of (color, top, bottom, left, right)
    curr = sub
    while True:
        h, w = len(curr), len(curr[0])
        if h <= 0 or w <= 0:
            break
        color = curr[0][0]
        if all(curr[r][c] == color for r in range(h) for c in range(w)):
            layers.append((color, None, None, None, None))  # center marker
            break

        top = 0
        for r in range(h):
            if all(curr[r][c] == color for c in range(w)):
                top += 1
            else:
                break

        bottom = 0
        for r in range(h - 1, -1, -1):
            if all(curr[r][c] == color for c in range(w)):
                bottom += 1
            else:
                break

        left = 0
        for c in range(w):
            if all(curr[r][c] == color for r in range(h)):
                left += 1
            else:
                break

        right = 0
        for c in range(w - 1, -1, -1):
            if all(curr[r][c] == color for r in range(h)):
                right += 1
            else:
                break

        layers.append((color, top, bottom, left, right))

        new_top = top
        new_bottom = h - bottom
        new_left = left
        new_right = w - right
        if new_top >= new_bottom or new_left >= new_right:
            break
        curr = [row[new_left:new_right] for row in curr[new_top:new_bottom]]

    # Apply the pattern to each target
    for tgt in targets:
        th, tw = tgt['h'], tgt['w']
        r0, c0 = tgt['rmin'], tgt['cmin']

        # Build the filled subgrid
        filled = [[0] * tw for _ in range(th)]

        # Current bounds within the target (local coords)
        cur_top, cur_left = 0, 0
        cur_bottom, cur_right = th, tw  # exclusive

        for color, t, b, l, r in layers:
            if cur_top >= cur_bottom or cur_left >= cur_right:
                break
            if t is None:  # center fill
                for rr in range(cur_top, cur_bottom):
                    for cc in range(cur_left, cur_right):
                        filled[rr][cc] = color
                break

            # Fill current layer borders
            # Top rows
            for rr in range(cur_top, min(cur_top + t, cur_bottom)):
                for cc in range(cur_left, cur_right):
                    filled[rr][cc] = color
            # Bottom rows
            for rr in range(max(cur_bottom - b, cur_top), cur_bottom):
                for cc in range(cur_left, cur_right):
                    filled[rr][cc] = color
            # Left cols
            for cc in range(cur_left, min(cur_left + l, cur_right)):
                for rr in range(cur_top, cur_bottom):
                    filled[rr][cc] = color
            # Right cols
            for cc in range(max(cur_right - r, cur_left), cur_right):
                for rr in range(cur_top, cur_bottom):
                    filled[rr][cc] = color

            cur_top += t
            cur_bottom -= b
            cur_left += l
            cur_right -= r

        # Write filled subgrid into result
        for rr in range(th):
            for cc in range(tw):
                result[r0 + rr][c0 + cc] = filled[rr][cc]

    return result
