"""
Solver for ARC-AGI task 8698868d.

The input grid contains:
  - A background color (most frequent)
  - A rectangular grid of same-sized "panels", each a single color with a few
    background-colored "dots" inside
  - Several same-sized "shapes" scattered on the background, each a colored
    rectangle with background-colored "holes" inside

The transformation:
  1. Match each shape to a panel by counting the connected components of holes
     in the shape's interior (4-connected) and matching to the panel whose dot
     count equals that component count.
  2. Embed each shape inside its matched panel (offset 1,1), replacing the
     shape's holes with the panel's own color.
  3. Output the panel grid with shapes embedded, stripping the background.
"""

from collections import Counter, deque, defaultdict


def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most frequent)
    color_counts = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    bg_color = color_counts.most_common(1)[0][0]

    # Find connected components of each non-background color
    visited = [[False] * cols for _ in range(rows)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg_color:
                color = grid[r][c]
                queue = deque([(r, c)])
                visited[r][c] = True
                min_r = max_r = r
                min_c = max_c = c
                while queue:
                    cr, cc = queue.popleft()
                    min_r, max_r = min(min_r, cr), max(max_r, cr)
                    min_c, max_c = min(min_c, cc), max(max_c, cc)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append({
                    'color': color,
                    'min_r': min_r, 'max_r': max_r,
                    'min_c': min_c, 'max_c': max_c,
                    'height': max_r - min_r + 1,
                    'width': max_c - min_c + 1,
                })

    # Separate panels (larger) from shapes (smaller) by bounding-box size
    size_groups = defaultdict(list)
    for comp in components:
        size_groups[(comp['height'], comp['width'])].append(comp)

    sizes = sorted(size_groups.keys(), key=lambda s: s[0] * s[1], reverse=True)
    panel_size = sizes[0]
    shape_size = sizes[1]
    panels = size_groups[panel_size]
    shapes = size_groups[shape_size]

    # Count background-colored dots inside each panel
    for panel in panels:
        panel['dot_count'] = sum(
            1 for r in range(panel['min_r'], panel['max_r'] + 1)
            for c in range(panel['min_c'], panel['max_c'] + 1)
            if grid[r][c] == bg_color
        )

    # Count connected components of background-colored holes in each shape interior
    for shape in shapes:
        ir0, ir1 = shape['min_r'] + 1, shape['max_r'] - 1
        ic0, ic1 = shape['min_c'] + 1, shape['max_c'] - 1
        seen = set()
        count = 0
        for r in range(ir0, ir1 + 1):
            for c in range(ic0, ic1 + 1):
                if grid[r][c] == bg_color and (r, c) not in seen:
                    q = deque([(r, c)])
                    seen.add((r, c))
                    while q:
                        cr, cc = q.popleft()
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if ir0 <= nr <= ir1 and ic0 <= nc <= ic1 and (nr, nc) not in seen and grid[nr][nc] == bg_color:
                                seen.add((nr, nc))
                                q.append((nr, nc))
                    count += 1
        shape['hole_components'] = count

    # Match: panel dot_count == shape hole_components
    dot_to_panel = {p['dot_count']: p for p in panels}
    hole_to_shape = {s['hole_components']: s for s in shapes}
    match_map = {}
    for cnt, panel in dot_to_panel.items():
        if cnt in hole_to_shape:
            match_map[(panel['min_r'], panel['min_c'])] = hole_to_shape[cnt]

    # Build the output grid
    panel_rows_set = sorted(set(p['min_r'] for p in panels))
    panel_cols_set = sorted(set(p['min_c'] for p in panels))
    ph, pw = panel_size
    output = [[0] * (len(panel_cols_set) * pw) for _ in range(len(panel_rows_set) * ph)]

    panel_map = {(p['min_r'], p['min_c']): p for p in panels}

    for gi, pr in enumerate(panel_rows_set):
        for gj, pc in enumerate(panel_cols_set):
            panel = panel_map.get((pr, pc))
            if panel is None:
                continue
            r0, c0 = gi * ph, gj * pw

            # Fill entire panel region with panel color
            for r in range(ph):
                for c in range(pw):
                    output[r0 + r][c0 + c] = panel['color']

            # Overlay matched shape at offset (1,1); holes become panel color
            shape = match_map.get((pr, pc))
            if shape:
                for sr in range(shape['height']):
                    for sc in range(shape['width']):
                        cell = grid[shape['min_r'] + sr][shape['min_c'] + sc]
                        if cell == bg_color:
                            output[r0 + 1 + sr][c0 + 1 + sc] = panel['color']
                        else:
                            output[r0 + 1 + sr][c0 + 1 + sc] = cell

    return output


if __name__ == '__main__':
    import json

    with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/8698868d.json') as f:
        task = json.load(f)

    all_pass = True
    for kind in ('train', 'test'):
        for i, pair in enumerate(task[kind]):
            result = solve(pair['input'])
            expected = pair['output']
            if result == expected:
                print(f"{kind.capitalize()} {i}: PASS")
            else:
                all_pass = False
                print(f"{kind.capitalize()} {i}: FAIL")
                print(f"  Expected {len(expected)}x{len(expected[0])}, got {len(result)}x{len(result[0])}")
                for r in range(min(len(expected), len(result))):
                    if expected[r] != result[r]:
                        print(f"  Row {r}: expected {expected[r]}")
                        print(f"         got      {result[r]}")

    if all_pass:
        print("\nAll pairs passed!")
