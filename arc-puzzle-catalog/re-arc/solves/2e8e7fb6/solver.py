from collections import Counter


def transform(grid):
    grid = [list(row) for row in grid]
    H, W = len(grid), len(grid[0])
    bg = Counter(c for row in grid for c in row).most_common(1)[0][0]

    color_cells = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != bg:
                color_cells.setdefault(v, []).append((r, c))

    def get_components(cells):
        cell_set = set(cells)
        visited = set()
        comps = []
        for cell in cells:
            if cell in visited:
                continue
            comp = [cell]
            visited.add(cell)
            stack = [cell]
            while stack:
                r, c = stack.pop()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = (r + dr, c + dc)
                    if nb in cell_set and nb not in visited:
                        visited.add(nb)
                        comp.append(nb)
                        stack.append(nb)
            comps.append(comp)
        return comps

    def find_center_and_frame(cells):
        if len(cells) == 1:
            return cells[0], []
        comps = get_components(cells)
        for comp in comps:
            if len(comp) != 1:
                continue
            cell = comp[0]
            other = [c for c in cells if c != cell]
            if not other:
                return cell, []
            rows = [r for r, c in other]
            cols = [c for r, c in other]
            r1, r2 = min(rows), max(rows)
            c1, c2 = min(cols), max(cols)
            cr, cc = cell
            if cr < r1 or cr > r2 or cc < c1 or cc > c2:
                return cell, other
        return None, cells

    def extract_gap_pattern(edge_vals, color, N):
        is_c = [1 if v == color else 0 for v in edge_vals]
        center = N // 2
        if is_c[center]:
            return [True] * N
        lo = hi = center
        while lo > 0 and not is_c[lo - 1]:
            lo -= 1
        while hi < N - 1 and not is_c[hi + 1]:
            hi += 1
        gap = hi - lo + 1
        k = (N - gap) // 2
        return [True] * k + [False] * gap + [True] * k

    def best_edge_pattern(r1, c1, r2, c2, color, N):
        h, w = r2 - r1 + 1, c2 - c1 + 1

        def try_col(ci, rstart):
            edge = [grid[r][ci] for r in range(rstart, rstart + N)]
            if edge[0] == color and edge[-1] == color:
                return sum(1 for v in edge if v == color), extract_gap_pattern(edge, color, N)
            return -1, None

        def try_row(ri, cstart):
            edge = [grid[ri][c] for c in range(cstart, cstart + N)]
            if edge[0] == color and edge[-1] == color:
                return sum(1 for v in edge if v == color), extract_gap_pattern(edge, color, N)
            return -1, None

        best_cnt, best_pat = -1, None
        if h == N:
            for ci in [c2, c1]:
                cnt, pat = try_col(ci, r1)
                if cnt > best_cnt:
                    best_cnt, best_pat = cnt, pat
        if w == N:
            for ri in [r2, r1]:
                cnt, pat = try_row(ri, c1)
                if cnt > best_cnt:
                    best_cnt, best_pat = cnt, pat

        return best_pat if best_pat else [True] * N

    frames = []
    center_color = None

    for color, cells in color_cells.items():
        cp, frame_cells = find_center_and_frame(cells)
        if cp is not None and center_color is None:
            center_color = color
        if not frame_cells:
            continue

        rows = [r for r, c in frame_cells]
        cols = [c for r, c in frame_cells]
        r1, r2 = min(rows), max(rows)
        c1, c2 = min(cols), max(cols)
        N = max(r2 - r1 + 1, c2 - c1 + 1)

        pat = best_edge_pattern(r1, c1, r2, c2, color, N)
        frames.append((color, N, pat))

    if not frames:
        return grid

    frames.sort(key=lambda x: -x[1])
    N_max = frames[0][1]

    out = [[bg] * N_max for _ in range(N_max)]
    for color, N, pattern in frames:
        L = (N_max - N) // 2
        for i in range(N):
            if pattern[i]:
                out[L][L + i] = color
                out[L + N - 1][L + i] = color
                out[L + i][L] = color
                out[L + i][L + N - 1] = color

    if center_color is not None:
        mid = N_max // 2
        out[mid][mid] = center_color

    return out
