from collections import Counter, deque


def find_ccs(grid, bg, connectivity=8):
    H, W = len(grid), len(grid[0])
    visited = [[False]*W for _ in range(H)]
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    ccs = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not visited[r][c]:
                cc = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc_ = q.popleft()
                    cc.append((cr, cc_, grid[cr][cc_]))
                    for dr, dc in dirs:
                        nr, nc = cr+dr, cc_+dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                ccs.append(cc)
    return ccs


def get_shape_rel(cc):
    min_r = min(r for r,c,_ in cc)
    min_c = min(c for _,c,_ in cc)
    return tuple(sorted((r-min_r, c-min_c) for r,c,_ in cc))


def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(c for row in grid for c in row).most_common(1)[0][0]
    output = [row[:] for row in grid]

    ccs_8 = find_ccs(grid, bg, connectivity=8)
    templates = [cc for cc in ccs_8 if len(set(c for _,_,c in cc)) >= 2]

    if templates:
        for tmpl in templates:
            min_r = min(r for r,c,_ in tmpl)
            min_c = min(c for _,c,_ in tmpl)
            pattern = {}
            for r, c, color in tmpl:
                pattern[(r-min_r, c-min_c)] = color

            for target_color in set(pattern.values()):
                sub_pos = sorted([(dr,dc) for (dr,dc),col in pattern.items() if col == target_color])
                other_entries = [(dr,dc,col) for (dr,dc),col in pattern.items() if col != target_color]
                if not sub_pos or not other_entries:
                    continue
                anchor = sub_pos[0]
                for gr in range(H):
                    for gc in range(W):
                        if grid[gr][gc] != target_color:
                            continue
                        off_r = gr - anchor[0]
                        off_c = gc - anchor[1]
                        matched = set()
                        all_match = True
                        for dr, dc in sub_pos:
                            nr, nc = dr + off_r, dc + off_c
                            if not (0 <= nr < H and 0 <= nc < W) or grid[nr][nc] != target_color:
                                all_match = False
                                break
                            matched.add((nr, nc))
                        if not all_match:
                            continue
                        if matched == {(r, c) for r, c, col in tmpl if col == target_color}:
                            continue
                        has_extra = False
                        for mr, mc in matched:
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = mr+dr, mc+dc
                                if (nr, nc) not in matched and 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == target_color:
                                    has_extra = True
                                    break
                            if has_extra:
                                break
                        if has_extra:
                            continue
                        has_missing = False
                        for dr, dc, col in other_entries:
                            nr, nc = dr + off_r, dc + off_c
                            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == bg:
                                has_missing = True
                                break
                        if not has_missing:
                            continue
                        for dr, dc, col in other_entries:
                            nr, nc = dr + off_r, dc + off_c
                            if 0 <= nr < H and 0 <= nc < W and output[nr][nc] == bg:
                                output[nr][nc] = col
    else:
        multi_ccs = [cc for cc in ccs_8 if len(cc) > 1]
        if not multi_ccs:
            return output

        shape_groups = {}
        for cc in multi_ccs:
            colors = set(c for _,_,c in cc)
            if len(colors) != 1:
                continue
            shape = get_shape_rel(cc)
            color = cc[0][2]
            if shape not in shape_groups:
                shape_groups[shape] = []
            origin = (min(r for r,c,_ in cc), min(c for _,c,_ in cc))
            shape_groups[shape].append((origin, color))

        for shape, instances in shape_groups.items():
            if len(instances) < 2:
                continue
            color = instances[0][1]
            origins = sorted([inst[0] for inst in instances])

            v1_col = origins[1][1] - origins[0][1]
            shape_cells = list(shape)
            max_dr = max(dr for dr, dc in shape_cells)
            max_dc = max(dc for dr, dc in shape_cells)

            col_incs = [v1_col, v1_col - 1]
            curr = origins[-1]
            step = 2

            while True:
                row_inc = step
                if step <= len(col_incs):
                    col_inc = col_incs[step - 1]
                else:
                    col_inc = col_incs[-2] + col_incs[-1] - 1
                    col_incs.append(col_inc)

                new_r = curr[0] + row_inc
                new_c = curr[1] + col_inc

                if new_r + max_dr >= H or new_c + max_dc >= W or new_c < 0 or new_r < 0:
                    break

                for dr, dc in shape_cells:
                    if output[new_r + dr][new_c + dc] == bg:
                        output[new_r + dr][new_c + dc] = color

                curr = (new_r, new_c)
                step += 1

    return output
