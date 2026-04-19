from collections import Counter


def get_bg(grid):
    return Counter(c for row in grid for c in row).most_common(1)[0][0]


def find_components(grid, bg):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    components = []

    def bfs(r, c):
        comp = {}
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited or cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if grid[cr][cc] == bg:
                continue
            visited.add((cr, cc))
            comp[(cr, cc)] = grid[cr][cc]
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    stack.append((cr + dr, cc + dc))
        return comp

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid[r][c] != bg:
                comp = bfs(r, c)
                if comp:
                    components.append(comp)
    return components


def is_rectangle(cells):
    if not cells:
        return False
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return len(cells) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def find_anchor_color(comps):
    candidates = Counter()
    total_2color = 0
    for comp in comps:
        colors = {}
        for (r, c), v in comp.items():
            colors.setdefault(v, []).append((r, c))
        if len(colors) != 2:
            continue
        total_2color += 1
        for color, cells in colors.items():
            if is_rectangle(cells):
                candidates[color] += 1
    for color, count in candidates.most_common():
        if count == total_2color:
            return color
    return candidates.most_common(1)[0][0] if candidates else None


def transform(grid):
    bg = get_bg(grid)
    rows, cols = len(grid), len(grid[0])
    comps = find_components(grid, bg)
    anchor_color = find_anchor_color(comps)

    template_tiles = set()
    for comp in comps:
        colors = {}
        for (r, c), v in comp.items():
            colors.setdefault(v, []).append((r, c))
        if len(colors) != 2 or anchor_color not in colors:
            continue
        anchor_cells = colors[anchor_color]
        fill_color_key = [k for k in colors if k != anchor_color][0]
        fill_cells = colors[fill_color_key]
        a_r0 = min(r for r, c in anchor_cells)
        a_c0 = min(c for r, c in anchor_cells)
        a_h = max(r for r, c in anchor_cells) - a_r0 + 1
        a_w = max(c for r, c in anchor_cells) - a_c0 + 1
        for r, c in fill_cells:
            tr = (r - a_r0) // a_h
            tc = (c - a_c0) // a_w
            template_tiles.add((tr, tc))

    result = [row[:] for row in grid]

    for comp in comps:
        colors = {}
        for (r, c), v in comp.items():
            colors.setdefault(v, []).append((r, c))

        if len(colors) == 2 and anchor_color in colors:
            anchor_cells = colors[anchor_color]
            fill_color = [k for k in colors if k != anchor_color][0]
            a_r0 = min(r for r, c in anchor_cells)
            a_c0 = min(c for r, c in anchor_cells)
            a_h = max(r for r, c in anchor_cells) - a_r0 + 1
            a_w = max(c for r, c in anchor_cells) - a_c0 + 1
            for tr, tc in template_tiles:
                for dr in range(a_h):
                    for dc in range(a_w):
                        nr = a_r0 + tr * a_h + dr
                        nc = a_c0 + tc * a_w + dc
                        if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == bg:
                            result[nr][nc] = fill_color

        elif len(colors) == 1:
            color = list(colors.keys())[0]
            cells = list(colors.values())[0]
            found = False
            for a_h in range(1, 5):
                for a_w in range(1, 5):
                    for a_r0 in range(min(r for r, c in cells), max(r for r, c in cells) + 1):
                        for a_c0 in range(min(c for r, c in cells), max(c for r, c in cells) + 1):
                            if not all(
                                (a_r0 + dr, a_c0 + dc) in comp
                                for dr in range(a_h) for dc in range(a_w)
                            ):
                                continue
                            all_valid = True
                            has_fill_tile = False
                            for r, c in cells:
                                tr = (r - a_r0) // a_h
                                tc = (c - a_c0) // a_w
                                if (tr, tc) == (0, 0):
                                    continue
                                if (tr, tc) not in template_tiles:
                                    all_valid = False
                                    break
                                has_fill_tile = True
                            if all_valid and has_fill_tile:
                                for tr, tc in template_tiles:
                                    for dr in range(a_h):
                                        for dc in range(a_w):
                                            nr = a_r0 + tr * a_h + dr
                                            nc = a_c0 + tc * a_w + dc
                                            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == bg:
                                                result[nr][nc] = color
                                for dr in range(a_h):
                                    for dc in range(a_w):
                                        nr = a_r0 + dr
                                        nc = a_c0 + dc
                                        if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == bg:
                                            result[nr][nc] = color
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

    return result
