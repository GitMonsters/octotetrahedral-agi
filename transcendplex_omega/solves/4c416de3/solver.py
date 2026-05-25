"""
ARC Task 4c416de3: Template shape expansion from markers

Pattern: A multi-cell template shape straddles a 0-bordered rectangle. Single-cell markers
inside 0-rectangles expand into oriented copies of the template shape, with the orientation
determined by which corner of the rectangle the marker is nearest to. The expansion crosses
through the 0-border, preserving the template's structural relationship to the border.

Two placement strategies based on template geometry:
- Templates with cells at bbox corners: tip-based anchor at the most isolated corner cell
- Templates without corner cells (Z/S shapes): elbow-based anchor at max-neighbor cells,
  selected by direction alignment scoring
"""


def solve(grid: list[list[int]]) -> list[list[int]]:
    from collections import Counter, deque

    rows, cols = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    dominant = Counter(flat).most_common(1)[0][0]

    color_cells: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != dominant and v != 0:
                color_cells.setdefault(v, []).append((r, c))

    DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def bfs8(cells_set: set, start: tuple) -> set:
        comp: set[tuple[int, int]] = set()
        q = deque([start])
        while q:
            r, c = q.popleft()
            if (r, c) in comp:
                continue
            comp.add((r, c))
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells_set and (nr, nc) not in comp:
                    q.append((nr, nc))
        return comp

    components: list[tuple[int, list]] = []
    markers: list[tuple[int, tuple[int, int]]] = []
    visited: set[tuple[int, int]] = set()
    for color, cells in color_cells.items():
        cells_set = set(cells)
        for cell in cells:
            if cell in visited:
                continue
            comp = bfs8(cells_set, cell)
            visited |= comp
            if len(comp) == 1:
                markers.append((color, list(comp)[0]))
            else:
                components.append((color, sorted(comp)))

    if not components:
        return grid

    _, template_cells = components[0]
    min_r = min(r for r, c in template_cells)
    min_c = min(c for r, c in template_cells)
    max_r = max(r for r, c in template_cells)
    max_c = max(c for r, c in template_cells)
    offsets = sorted((r - min_r, c - min_c) for r, c in template_cells)
    h = max_r - min_r
    w = max_c - min_c
    offsets_set = set(offsets)

    bbox_corners = [(0, 0), (0, w), (h, 0), (h, w)]
    corner_cells = [(cr, cc) for cr, cc in bbox_corners if (cr, cc) in offsets_set]

    output = [row[:] for row in grid]

    def nearest_zeros(mr: int, mc: int) -> tuple:
        dist_up = float('inf')
        for r in range(mr - 1, -1, -1):
            if grid[r][mc] == 0:
                dist_up = mr - r
                break
        dist_down = float('inf')
        for r in range(mr + 1, rows):
            if grid[r][mc] == 0:
                dist_down = r - mr
                break
        dist_left = float('inf')
        for c in range(mc - 1, -1, -1):
            if grid[mr][c] == 0:
                dist_left = mc - c
                break
        dist_right = float('inf')
        for c in range(mc + 1, cols):
            if grid[mr][c] == 0:
                dist_right = c - mc
                break
        return dist_up, dist_down, dist_left, dist_right

    if corner_cells:
        # Strategy 1: tip at the most isolated corner cell
        tip = min(corner_cells,
                  key=lambda cc: sum(1 for dr, dc in DIRS8
                                     if (cc[0] + dr, cc[1] + dc) in offsets_set))

        orients = {
            'orig': (offsets, tip),
            'hflip': (sorted((r, w - c) for r, c in offsets), (tip[0], w - tip[1])),
            'vflip': (sorted((h - r, c) for r, c in offsets), (h - tip[0], tip[1])),
            'rot180': (sorted((h - r, w - c) for r, c in offsets), (h - tip[0], w - tip[1])),
        }
        tip_to_orient = {t: offs for _, (offs, t) in orients.items()}

        for mcolor, (mr, mc) in markers:
            du, dd, dl, dr_ = nearest_zeros(mr, mc)
            go_up = du <= dd
            go_left = dl <= dr_

            if go_up and go_left:
                needed_tip = (h, w)
            elif go_up and not go_left:
                needed_tip = (h, 0)
            elif not go_up and go_left:
                needed_tip = (0, w)
            else:
                needed_tip = (0, 0)

            if needed_tip in tip_to_orient:
                orient_offs = tip_to_orient[needed_tip]
                tip_r, tip_c = needed_tip
                for or_r, or_c in orient_offs:
                    pr = mr + (or_r - tip_r)
                    pc = mc + (or_c - tip_c)
                    if 0 <= pr < rows and 0 <= pc < cols:
                        output[pr][pc] = mcolor

    else:
        # Strategy 2: elbow-based with direction scoring
        all_orients = [
            offsets,
            sorted((r, w - c) for r, c in offsets),
            sorted((h - r, c) for r, c in offsets),
            sorted((h - r, w - c) for r, c in offsets),
        ]
        unique_orients: list[list] = []
        seen: set[tuple] = set()
        for o in all_orients:
            key = tuple(o)
            if key not in seen:
                seen.add(key)
                unique_orients.append(o)

        def find_elbows(offs: list) -> list:
            offs_set = set(offs)
            counts = {}
            for r, c in offs:
                n = sum(1 for dr, dc in DIRS8 if (r + dr, c + dc) in offs_set)
                counts[(r, c)] = n
            max_n = max(counts.values())
            return [k for k, v in counts.items() if v == max_n]

        for mcolor, (mr, mc) in markers:
            du, dd, dl, dr_ = nearest_zeros(mr, mc)
            sign_r = -1 if du <= dd else 1
            sign_c = -1 if dl <= dr_ else 1

            best = None
            best_score = -float('inf')

            for orient in unique_orients:
                elbows = find_elbows(orient)
                for anchor in elbows:
                    placed = [(mr + or_r - anchor[0], mc + or_c - anchor[1])
                              for or_r, or_c in orient]

                    valid = True
                    crosses_zero = False
                    for pr, pc in placed:
                        if not (0 <= pr < rows and 0 <= pc < cols):
                            valid = False
                            break
                        v = grid[pr][pc]
                        if v == 0:
                            crosses_zero = True
                        elif v != dominant and (pr, pc) != (mr, mc):
                            valid = False
                            break

                    if valid and crosses_zero:
                        cent_r = sum(pr for pr, pc in placed) / len(placed) - mr
                        cent_c = sum(pc for pr, pc in placed) / len(placed) - mc
                        score = sign_r * cent_r + sign_c * cent_c
                        if score > best_score:
                            best_score = score
                            best = placed

            if best:
                for pr, pc in best:
                    output[pr][pc] = mcolor

    return output


if __name__ == "__main__":
    import sys
    import json

    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task.get('test', [])):
        result = solve(ex['input'])
        is_train = i < len(task['train'])
        idx = i if is_train else i - len(task['train'])
        label = 'Train' if is_train else 'Test'
        if 'output' in ex:
            status = "PASS" if result == ex['output'] else "FAIL"
        else:
            status = "computed"
        print(f"{label} {idx}: {status}")
