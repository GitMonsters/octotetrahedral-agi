def transform(grid):
    from collections import Counter
    from itertools import combinations

    H, W = len(grid), len(grid[0])
    counts = Counter(v for row in grid for v in row)
    bg = counts.most_common(1)[0][0]

    visited = [[False] * W for _ in range(H)]
    components = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                queue = [(r, c)]
                visited[r][c] = True
                cells = [(r, c)]
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                            cells.append((nr, nc))
                components.append((color, cells))

    by_color = {}
    for color, cells in components:
        by_color.setdefault(color, []).append(cells)

    valid_sizes = [9, 7, 5, 3, 1]
    rings = []

    for color, comp_list in by_color.items():
        used = [False] * len(comp_list)
        for target_size in valid_sizes:
            while True:
                unused = [i for i in range(len(comp_list)) if not used[i]]
                if not unused:
                    break
                found = False
                for r_count in range(len(unused), 0, -1):
                    if found:
                        break
                    for combo in combinations(unused, r_count):
                        all_cells = []
                        for i in combo:
                            all_cells.extend(comp_list[i])
                        min_r = min(r for r, c in all_cells)
                        max_r = max(r for r, c in all_cells)
                        min_c = min(c for r, c in all_cells)
                        max_c = max(c for r, c in all_cells)
                        mx = max(max_r - min_r + 1, max_c - min_c + 1)
                        if mx == target_size:
                            tmpl = [[bg] * mx for _ in range(mx)]
                            for r, c in all_cells:
                                tmpl[r - min_r][c - min_c] = color
                            rings.append((mx, color, tmpl))
                            for i in combo:
                                used[i] = True
                            found = True
                            break
                if not found:
                    break

    def make_sym(tmpl, N, bg_val, col):
        full = [[bg_val] * N for _ in range(N)]
        for r in range(N):
            for c in range(N):
                if tmpl[r][c] != bg_val:
                    full[r][c] = col
                    full[r][N - 1 - c] = col
                    full[N - 1 - r][c] = col
                    full[N - 1 - r][N - 1 - c] = col
        return full

    out = [[bg] * 9 for _ in range(9)]
    rings.sort(key=lambda x: -x[0])

    for ring_size, color, tmpl in rings:
        sym = make_sym(tmpl, ring_size, bg, color)
        offset = (9 - ring_size) // 2
        for r in range(ring_size):
            for c in range(ring_size):
                if sym[r][c] != bg:
                    out[offset + r][offset + c] = sym[r][c]
    return out
