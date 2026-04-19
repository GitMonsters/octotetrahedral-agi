"""
Solver for ARC puzzle 3ede399d.
- 3+ colors: third color is noise, replace with majority surrounding color
- 2 colors balanced: protrusions removed via local majority vote
- 2 colors dominant: shapes reflected about rotation center
"""
from collections import deque, Counter


def find_components(grid, color):
    R, C = len(grid), len(grid[0])
    visited = set()
    components = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] == color and (r, c) not in visited:
                comp = set()
                q = deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    comp.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < R and 0 <= nc < C and
                                (nr, nc) not in visited and grid[nr][nc] == color):
                            visited.add((nr, nc))
                            q.append((nr, nc))
                components.append(comp)
    return components


def get_surrounding_majority(grid, cells, noise_colors):
    """Find the most common non-noise color surrounding a set of cells."""
    R, C = len(grid), len(grid[0])
    border_colors = Counter()
    for r, c in cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in cells:
                if grid[nr][nc] not in noise_colors:
                    border_colors[grid[nr][nc]] += 1
    if border_colors:
        return border_colors.most_common(1)[0][0]
    # Fallback: BFS outward
    visited = set(cells)
    q = deque(list(cells))
    while q:
        cr, cc = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in visited:
                if grid[nr][nc] not in noise_colors:
                    return grid[nr][nc]
                visited.add((nr, nc))
                q.append((nr, nc))
    cnt = Counter()
    for row in grid:
        cnt.update(row)
    for c in noise_colors:
        if c in cnt:
            del cnt[c]
    return cnt.most_common(1)[0][0] if cnt else grid[0][0]


def noise_removal_3color(grid):
    R, C = len(grid), len(grid[0])
    color_counts = Counter()
    for row in grid:
        color_counts.update(row)
    sorted_colors = color_counts.most_common()
    region_colors = {sorted_colors[0][0], sorted_colors[1][0]}
    noise_colors = set(color_counts.keys()) - region_colors
    output = [row[:] for row in grid]

    # Find connected components of noise colors
    noise_cells = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] in noise_colors:
                noise_cells.add((r, c))

    visited = set()
    for r, c in noise_cells:
        if (r, c) not in visited:
            comp = set()
            q = deque([(r, c)])
            visited.add((r, c))
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < R and 0 <= nc < C and
                            (nr, nc) not in visited and (nr, nc) in noise_cells):
                        visited.add((nr, nc))
                        q.append((nr, nc))
            fill_color = get_surrounding_majority(grid, comp, noise_colors)
            for cr, cc in comp:
                output[cr][cc] = fill_color

    return output


def noise_removal_2color(grid):
    """Remove protrusions using local majority vote in a 5x5 window."""
    R, C = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    rad = 2

    for r in range(R):
        for c in range(C):
            cnt = Counter()
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C:
                        cnt[grid[nr][nc]] += 1
            majority = cnt.most_common(1)[0][0]
            if grid[r][c] != majority:
                output[r][c] = majority
    return output


def shape_reflection(grid):
    R, C = len(grid), len(grid[0])
    color_counts = Counter()
    for row in grid:
        color_counts.update(row)
    sorted_colors = color_counts.most_common()
    bg = sorted_colors[0][0]
    fg = sorted_colors[1][0]

    fg_comps = find_components(grid, fg)
    fg_comps.sort(key=lambda x: -len(x))
    if not fg_comps:
        return [row[:] for row in grid]

    largest = fg_comps[0]
    pr = (R - 1) / 2
    pc = max(c for _, c in largest)

    lrg_max_r = max(r for r, _ in largest)
    lrg_min_r = min(r for r, _ in largest)
    lrg_center_r = (lrg_min_r + lrg_max_r) / 2
    lrg_height = lrg_max_r - lrg_min_r + 1
    lrg_new_center_r = 2 * pr - lrg_center_r
    lrg_new_max_r = round(lrg_new_center_r + (lrg_height - 1) / 2)

    output = [[bg] * C for _ in range(R)]

    for comp in fg_comps:
        min_r = min(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_r = max(r for r, c in comp)
        max_c = max(c for r, c in comp)
        height = max_r - min_r + 1
        width = max_c - min_c + 1

        new_center_r = 2 * pr - (min_r + max_r) / 2
        new_center_c = 2 * pc - (min_c + max_c) / 2
        new_min_r = round(new_center_r - (height - 1) / 2)
        new_min_c = round(new_center_c - (width - 1) / 2)
        new_max_c = new_min_c + width - 1

        if new_max_c >= C - 2:
            bounced_max_c = int(2 * pc - new_max_c)
            new_min_c = bounced_max_c - width + 1
            dist = min_r - lrg_max_r
            new_min_r = lrg_new_max_r - dist - height + 1

        if new_min_c < 2 and new_min_c < min_c:
            bounced_min_c = int(2 * pc - new_min_c)
            new_min_c = bounced_min_c
            dist = min_r - lrg_max_r
            new_min_r = lrg_new_max_r - dist - height + 1

        for r, c in comp:
            nr = new_min_r + r - min_r
            nc = new_min_c + c - min_c
            if 0 <= nr < R and 0 <= nc < C:
                output[nr][nc] = fg

    return output


def transform(input_grid: list) -> list:
    R = len(input_grid)
    C = len(input_grid[0])
    color_counts = Counter()
    for row in input_grid:
        color_counts.update(row)
    all_colors = set(color_counts.keys())
    sorted_colors = color_counts.most_common()

    if len(all_colors) >= 3:
        return noise_removal_3color(input_grid)

    if len(all_colors) == 2:
        total = R * C
        dominant_ratio = sorted_colors[0][1] / total
        if dominant_ratio > 0.85:
            return shape_reflection(input_grid)
        else:
            return noise_removal_2color(input_grid)

    return [row[:] for row in input_grid]
