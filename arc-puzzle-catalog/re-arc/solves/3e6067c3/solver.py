from typing import List


def transform(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # 1. Find the sequence row (bottom-up scan):
    #    The sequence row contains non-1/non-8 values spaced exactly 2 columns apart.
    sequence = []
    seq_row  = -1
    for r in range(rows - 1, -1, -1):
        non8 = [(c, grid[r][c]) for c in range(cols) if grid[r][c] != 8]
        if len(non8) >= 2:
            vals = [v for _, v in non8]
            cs   = [c for c, _ in non8]
            if all(v != 1 for v in vals):
                diffs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
                if all(d == 2 for d in diffs):
                    sequence = vals
                    seq_row  = r
                    break

    # 2. Find all colored boxes (non-8, non-1 interior regions, excluding seq_row).
    #    Each box is a rectangle of 1s with a colored interior cell (or cells).
    visited = set()
    boxes   = []

    for r in range(rows):
        if r == seq_row:
            continue
        for c in range(cols):
            v = grid[r][c]
            if v != 8 and v != 1 and (r, c) not in visited:
                color  = v
                region = []
                stack  = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in visited or cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if grid[cr][cc] == color:
                        visited.add((cr, cc))
                        region.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((cr + dr, cc + dc))

                # Determine outer boundary by scanning outward until hitting 8.
                r0, c0 = region[0]
                ot = r0
                while ot > 0 and grid[ot - 1][c0] != 8:
                    ot -= 1
                ob = r0
                while ob < rows - 1 and grid[ob + 1][c0] != 8:
                    ob += 1
                ol = c0
                while ol > 0 and grid[r0][ol - 1] != 8:
                    ol -= 1
                or_ = c0
                while or_ < cols - 1 and grid[r0][or_ + 1] != 8:
                    or_ += 1

                ir = (min(rr for rr, _ in region), max(rr for rr, _ in region))
                ic = (min(cc for _, cc in region), max(cc for _, cc in region))

                boxes.append({
                    'color': color,
                    'ir': ir, 'ic': ic,
                    'ot': ot, 'ob': ob,
                    'ol': ol, 'or': or_,
                })

    # 3. Trace the Hamiltonian path through boxes using the sequence.
    #    Two boxes are adjacent if they share the same row-band (same outer-top)
    #    or the same column-band (same outer-left and outer-right).
    def adjacent(a, b):
        return (a['ot'] == b['ot'] or
                (a['ol'] == b['ol'] and a['or'] == b['or']))

    used = [False] * len(boxes)
    path = []

    def trace(idx, step):
        if step == len(sequence):
            return True
        cur = boxes[idx]
        for i, box in enumerate(boxes):
            if not used[i] and box['color'] == sequence[step] and adjacent(cur, box):
                used[i] = True
                path.append(i)
                if trace(i, step + 1):
                    return True
                path.pop()
                used[i] = False
        return False

    for i, box in enumerate(boxes):
        if box['color'] == sequence[0]:
            used[i] = True
            path.append(i)
            if trace(i, 1):
                break
            path.pop()
            used[i] = False

    # 4. Draw the gap connections.
    #    For each consecutive pair A->B in the path, fill the gap between them
    #    with A's color, at A's inner rows (horizontal) or inner cols (vertical).
    result = [row[:] for row in grid]

    for step in range(len(path) - 1):
        a     = boxes[path[step]]
        b     = boxes[path[step + 1]]
        color = a['color']

        if a['ot'] == b['ot']:   # horizontal connection
            lo, hi = (a, b) if a['ol'] < b['ol'] else (b, a)
            for c in range(lo['or'] + 1, hi['ol']):
                for r in range(a['ir'][0], a['ir'][1] + 1):
                    result[r][c] = color
        else:                    # vertical connection
            top, bot = (a, b) if a['ot'] < b['ot'] else (b, a)
            for r in range(top['ob'] + 1, bot['ot']):
                for c in range(a['ic'][0], a['ic'][1] + 1):
                    result[r][c] = color

    return result
