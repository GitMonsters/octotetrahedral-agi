"""
b0039139 — Template Tiling

Input has 4 sections separated by 3 lines of a separator color:
  Section 0: Template pattern (bordered with bg, interior = binary pattern)
  Section 1: Mark cells (count / 2 = number of tiles)
  Section 2: Solid color for "filled" template cells
  Section 3: Solid color for "empty" template cells and gaps

Output tiles the template N times with 1-cell gaps of color2 between tiles.
Direction: vertical if row separators, horizontal if column separators.
"""

def solve(input_grid):
    H = len(input_grid)
    W = len(input_grid[0])

    sep_rows = []
    sep_cols = []
    for color in range(10):
        rows = [r for r in range(H) if all(input_grid[r][c] == color for c in range(W))]
        cols = [c for c in range(W) if all(input_grid[r][c] == color for r in range(H))]
        if len(rows) == 3:
            sep_rows = rows
            break
        if len(cols) == 3:
            sep_cols = cols
            break

    if sep_rows:
        boundaries = [-1] + sep_rows + [H]
        sections = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i] + 1, boundaries[i + 1]
            if s < e:
                sections.append([list(input_grid[r]) for r in range(s, e)])
        tiling_dir = 'vertical'
    else:
        boundaries = [-1] + sep_cols + [W]
        sections = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i] + 1, boundaries[i + 1]
            if s < e:
                sections.append([[input_grid[r][c] for c in range(s, e)] for r in range(H)])
        tiling_dir = 'horizontal'

    ts = sections[0]
    template = []
    for r in range(1, len(ts) - 1):
        template.append([1 if ts[r][c] != 0 else 0 for c in range(1, len(ts[0]) - 1)])
    th, tw = len(template), len(template[0])

    mark_count = sum(1 for r in sections[1] for v in r if v != 0)
    tile_count = mark_count // 2

    color1 = sections[2][0][0]
    color2 = sections[3][0][0]

    tile = [[color1 if template[r][c] else color2 for c in range(tw)] for r in range(th)]

    if tiling_dir == 'vertical':
        output = []
        for t in range(tile_count):
            if t > 0:
                output.append([color2] * tw)
            for r in range(th):
                output.append(list(tile[r]))
    else:
        output = []
        for r in range(th):
            row = []
            for t in range(tile_count):
                if t > 0:
                    row.append(color2)
                row.extend(tile[r])
            output.append(row)

    return output
