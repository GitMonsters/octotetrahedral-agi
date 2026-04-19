from collections import Counter


def transform(grid):
    h, w = len(grid), len(grid[0])
    out = [[0] * (w * 3) for _ in range(h * 3)]

    counts = Counter(value for row in grid for value in row)
    bg = counts.most_common(1)[0][0]
    for r in range(h * 3):
        for c in range(w * 3):
            out[r][c] = bg

    shapes = []
    visited = set()
    for r in range(h - 1):
        for c in range(w - 1):
            if (r, c) in visited:
                continue
            cells = {}
            valid = True
            for dr in range(2):
                for dc in range(2):
                    rr, cc = r + dr, c + dc
                    value = grid[rr][cc]
                    if value == bg:
                        valid = False
                        break
                    cells[(rr, cc)] = value
                if not valid:
                    break
            if not valid or len(cells) != 4:
                continue

            color_counts = Counter(cells.values())
            if len(color_counts) != 2:
                continue

            marker = fill = None
            for color, count in color_counts.items():
                if count == 1:
                    marker = color
                elif count == 3:
                    fill = color
            if marker is None or fill is None:
                continue

            marker_pos = next(pos for pos, value in cells.items() if value == marker)
            fill_positions = [pos for pos, value in cells.items() if value == fill]
            visited.update(cells)
            shapes.append(
                {
                    "marker": marker_pos,
                    "fills": fill_positions,
                    "fill_color": fill,
                }
            )

    if len(shapes) < 2:
        return out

    fill_color = shapes[0]["fill_color"]
    pair_index = 0
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            a = (shapes[i], shapes[j]["marker"], pair_index * 2)
            b = (shapes[j], shapes[i]["marker"], pair_index * 2 + 1)
            for shape, other_marker, idx in (a, b):
                marker_pos = shape["marker"]
                for rr, cc in shape["fills"]:
                    dr = rr - marker_pos[0]
                    dc = cc - marker_pos[1]
                    if dr != 0 and dc != 0:
                        continue
                    row_off = (6 if dr != 0 else 3) + idx * 4
                    col_off = 3 + idx * 4
                    br = rr + other_marker[0] + row_off
                    bc = cc + other_marker[1] + col_off
                    for r2 in range(br, br + 4):
                        for c2 in range(bc, bc + 4):
                            if 0 <= r2 < h * 3 and 0 <= c2 < w * 3:
                                out[r2][c2] = fill_color

            pair_index += 1

    return out
