from collections import Counter


def _rotate90(r, c, sr, sc):
    return (sr + 2 * c - sc) // 2, (sc - 2 * r + sr) // 2


def _find_center(grid, bg):
    h, w = len(grid), len(grid[0])
    points = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] != bg]
    best_score = None
    best_center = (0, 0)

    for sr in range(2 * h - 1):
        for sc in range(2 * w - 1):
            if (sr - sc) % 2:
                continue

            conflicts = 0
            matches = 0
            in_bounds = True

            for r, c, value in points:
                rr, cc = r, c
                for _ in range(3):
                    nr_num = sr + 2 * cc - sc
                    nc_num = sc - 2 * rr + sr
                    if nr_num % 2 or nc_num % 2:
                        in_bounds = False
                        conflicts = 1
                        break
                    rr, cc = nr_num // 2, nc_num // 2
                    if not (0 <= rr < h and 0 <= cc < w):
                        in_bounds = False
                        conflicts = 1
                        break
                    cell = grid[rr][cc]
                    if cell == bg:
                        continue
                    if cell != value:
                        conflicts = 1
                        break
                    matches += 1
                if conflicts:
                    break

            score = (conflicts, 0 if in_bounds else 1, -matches)
            if best_score is None or score < best_score:
                best_score = score
                best_center = (sr, sc)

    return best_center


def transform(grid):
    bg = Counter(value for row in grid for value in row).most_common(1)[0][0]
    sr, sc = _find_center(grid, bg)
    out = [row[:] for row in grid]

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == bg:
                continue
            rr, cc = r, c
            for _ in range(3):
                rr, cc = _rotate90(rr, cc, sr, sc)
                out[rr][cc] = value

    return out
