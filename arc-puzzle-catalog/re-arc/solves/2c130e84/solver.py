"""
Solver for ARC task 2c130e84

Rule:
1. Find the rectangular frame (hollow rectangle of a single non-bg color).
2. Extract the interior pattern.
3. If the interior has non-background content, use its bounding box as the
   template size and its pixel pattern as the stamp.
4. If the interior is all background, use the full interior dimensions as the
   template. If no shapes match, shrink by peeling one border layer
   (max(h-2,1) x max(w-2,1)) and retry.
5. Replace all connected components whose bounding box matches the template
   by stamping the pattern over their bounding box area.
"""


def transform(grid):
    grid = [list(row) for row in grid]
    rows, cols = len(grid), len(grid[0])
    flat = [v for row in grid for v in row]
    bg = max(set(flat), key=flat.count)

    # Find rectangular frame (hollow rectangle of single non-bg color)
    frame = None
    for r1 in range(rows):
        for c1 in range(cols):
            if grid[r1][c1] == bg:
                continue
            bc = grid[r1][c1]
            for r2 in range(r1 + 2, rows):
                for c2 in range(c1 + 2, cols):
                    ok = True
                    for c in range(c1, c2 + 1):
                        if grid[r1][c] != bc or grid[r2][c] != bc:
                            ok = False
                            break
                    if not ok:
                        continue
                    for r in range(r1, r2 + 1):
                        if grid[r][c1] != bc or grid[r][c2] != bc:
                            ok = False
                            break
                    if not ok:
                        continue
                    for r in range(r1 + 1, r2):
                        for c in range(c1 + 1, c2):
                            if grid[r][c] == bc:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    frame = (r1, c1, r2, c2, bc)
                    break
                if frame:
                    break
            if frame:
                break
        if frame:
            break

    r1, c1, r2, c2, bc = frame
    int_h = r2 - r1 - 1
    int_w = c2 - c1 - 1

    # Extract interior
    interior = [
        [grid[r][c] for c in range(c1 + 1, c2)] for r in range(r1 + 1, r2)
    ]

    # Find non-bg content in interior
    non_bg = [
        (r, c)
        for r in range(int_h)
        for c in range(int_w)
        if interior[r][c] != bg
    ]

    if non_bg:
        nb_r1 = min(r for r, c in non_bg)
        nb_r2 = max(r for r, c in non_bg)
        nb_c1 = min(c for r, c in non_bg)
        nb_c2 = max(c for r, c in non_bg)
        template_h = nb_r2 - nb_r1 + 1
        template_w = nb_c2 - nb_c1 + 1
        stamp = [
            [interior[r][c] for c in range(nb_c1, nb_c2 + 1)]
            for r in range(nb_r1, nb_r2 + 1)
        ]
        has_content = True
    else:
        template_h = int_h
        template_w = int_w
        stamp = [[bg] * template_w for _ in range(template_h)]
        has_content = False

    # Exclude frame region from component search
    frame_region = set()
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            frame_region.add((r, c))

    # Find connected components outside frame
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if (
                grid[r][c] != bg
                and (r, c) not in visited
                and (r, c) not in frame_region
            ):
                comp = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and (nr, nc) not in visited
                            and grid[nr][nc] != bg
                            and (nr, nc) not in frame_region
                        ):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                components.append(comp)

    # Find matches by bounding box
    def find_matches(th, tw):
        matches = []
        for comp in components:
            min_r = min(r for r, c in comp)
            max_r = max(r for r, c in comp)
            min_c = min(c for r, c in comp)
            max_c = max(c for r, c in comp)
            if max_r - min_r + 1 == th and max_c - min_c + 1 == tw:
                matches.append((min_r, min_c))
        return matches

    matches = find_matches(template_h, template_w)

    # If no match and interior was all bg, shrink template by peeling layers
    if not matches and not has_content:
        th, tw = template_h, template_w
        while not matches:
            th = max(th - 2, 1)
            tw = max(tw - 2, 1)
            if th == 1 and tw == 1:
                break
            stamp = [[bg] * tw for _ in range(th)]
            matches = find_matches(th, tw)

    # Apply stamp over matched bounding boxes
    result = [list(row) for row in grid]
    sh = len(stamp)
    sw = len(stamp[0])
    for mr, mc in matches:
        for dr in range(sh):
            for dc in range(sw):
                result[mr + dr][mc + dc] = stamp[dr][dc]

    return result
