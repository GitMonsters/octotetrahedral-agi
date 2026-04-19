from copy import deepcopy


def get_bg(grid):
    flat = [c for row in grid for c in row]
    return max(set(flat), key=flat.count)


def get_components(grid, bg):
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                comp = []
                stack = [(r, c)]
                while stack:
                    nr, nc = stack.pop()
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != bg:
                        visited[nr][nc] = True
                        comp.append((nr, nc, grid[nr][nc]))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((nr + dr, nc + dc))
                components.append(comp)
    return components


def comp_tl(comp):
    return (min(c[0] for c in comp), min(c[1] for c in comp))


def normalize(comp):
    tl = comp_tl(comp)
    return [(r - tl[0], c - tl[1], v) for r, c, v in comp]


def hflip(cells):
    max_c = max(c for r, c, v in cells)
    return [(r, max_c - c, v) for r, c, v in cells]


def place_copy(out, cells, tl, rows, cols):
    for dr, dc, v in cells:
        nr, nc = tl[0] + dr, tl[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out[nr][nc] = v


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = get_bg(grid)

    comps = get_components(grid, bg)
    isolates = [(c[0][0], c[0][1], c[0][2]) for c in comps if len(c) == 1]
    templates = [c for c in comps if len(c) > 1]

    out = deepcopy(grid)

    if not templates or not isolates:
        return out

    # For each template, build normalized cell list and hflip version
    template_data = []
    for tmpl in templates:
        cells_norm = normalize(tmpl)
        cells_hflip = hflip(cells_norm)
        # Renormalize hflip
        hfl_tl = (min(r for r, c, v in cells_hflip), min(c for r, c, v in cells_hflip))
        cells_hflip_norm = [(r - hfl_tl[0], c - hfl_tl[1], v) for r, c, v in cells_hflip]
        template_data.append({
            'orig': cells_norm,
            'hflip': cells_hflip_norm,
            'tl': comp_tl(tmpl),
        })

    def find_template_for_color(color, orientation):
        """Find the template containing `color`, return (tmpl_idx, cells, anchor_offset)"""
        for tidx, td in enumerate(template_data):
            cells = td[orientation]
            # Find cells with matching color
            matching = [(r, c) for r, c, v in cells if v == color]
            if matching:
                # Use the topmost-leftmost match
                anchor = min(matching)
                return tidx, cells, anchor
        return None, None, None

    for iso_r, iso_c, iso_v in isolates:
        if iso_v == 7:
            # Isolated 7 → hflip orientation
            tidx, cells, anchor = find_template_for_color(7, 'hflip')
            if tidx is None:
                continue
            tl = (iso_r - anchor[0], iso_c - anchor[1])
        else:
            # Other colors → original orientation
            tidx, cells, anchor = find_template_for_color(iso_v, 'orig')
            if tidx is None:
                continue
            tl = (iso_r - anchor[0], iso_c - anchor[1])

        place_copy(out, cells, tl, rows, cols)

    return out


if __name__ == "__main__":
    import json

    with open('/Users/evanpieser/Desktop/ReArc45/re-arc_test_challenges-2026-04-05T23-26-25.json') as f:
        data = json.load(f)

    task = data['6a7a550f']
    passed = 0
    for i, pair in enumerate(task['train']):
        inp = pair['input']
        expected = pair['output']
        got = transform(inp)
        if got == expected:
            print(f"Pair {i}: PASS")
            passed += 1
        else:
            print(f"Pair {i}: FAIL")
            # Show differences
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if got[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): expected={expected[r][c]}, got={got[r][c]}")
    print(f"Passed {passed}/{len(task['train'])}")
