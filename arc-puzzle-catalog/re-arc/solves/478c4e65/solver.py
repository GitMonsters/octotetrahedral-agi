from collections import Counter
from typing import List, Set, Dict, Tuple
import math

def transform(input_grid: List[List[int]]) -> List[List[int]]:
    R, C = len(input_grid), len(input_grid[0])
    grid = [row[:] for row in input_grid]
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    non_bg = {}
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg:
                non_bg[(r,c)] = grid[r][c]

    parent = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[a] = b
    for pos in non_bg: parent[pos] = pos
    for (r,c) in non_bg:
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==0 and dc==0: continue
                nb = (r+dr, c+dc)
                if nb in non_bg: union((r,c), nb)
    components = {}
    for pos in non_bg:
        root = find(pos)
        components.setdefault(root, []).append(pos)

    def do_reflect(body_cells, mpos, bpos, exclude_minority=True, force_exclude_side=None):
        mr, mc = mpos; br, bc = bpos
        dr, dc = mr-br, mc-bc
        is_4adj = abs(dr)+abs(dc)==1
        results = {}
        if is_4adj:
            perp_counts = {}
            for (r2,c2) in body_cells:
                if dr != 0:
                    side = c2 - bc
                else:
                    side = r2 - br
                if side != 0:
                    key = 1 if side > 0 else -1
                    perp_counts[key] = perp_counts.get(key, 0) + 1
            count_pos = perp_counts.get(1, 0)
            count_neg = perp_counts.get(-1, 0)
            minority_side = force_exclude_side
            if minority_side is None and exclude_minority and count_pos > 0 and count_neg > 0 and count_pos != count_neg:
                minority_side = 1 if count_pos < count_neg else -1
            
            for (r,c) in body_cells:
                if dr != 0:
                    if dr>0 and r>=mr: continue
                    if dr<0 and r<=mr: continue
                    if minority_side is not None:
                        if abs(r-mr)==1 and abs(c-mc)==1:
                            perp_val = c - bc
                            if (perp_val > 0 and minority_side == 1) or (perp_val < 0 and minority_side == -1):
                                continue
                    nr, nc = mr+br-r, c
                else:
                    if dc>0 and c>=mc: continue
                    if dc<0 and c<=mc: continue
                    if minority_side is not None:
                        if abs(r-mr)==1 and abs(c-mc)==1:
                            perp_val = r - br
                            if (perp_val > 0 and minority_side == 1) or (perp_val < 0 and minority_side == -1):
                                continue
                    nr, nc = r, mc+bc-c
                if 0<=nr<R and 0<=nc<C and (nr,nc)!=mpos:
                    results[(nr,nc)] = True
        else:
            mid_r, mid_c = (mr+br)/2, (mc+bc)/2
            dir_r, dir_c = br-mid_r, bc-mid_c
            for (r,c) in body_cells:
                if abs(r-mr)+abs(c-mc)==1: continue
                dot = (r-mid_r)*dir_r + (c-mid_c)*dir_c
                if dot <= 0: continue
                nr, nc = mr+br-r, mc+bc-c
                if 0<=nr<R and 0<=nc<C and (nr,nc)!=mpos:
                    results[(nr,nc)] = True
        return results

    def do_step2(new_cells, mpos, bpos):
        mr, mc = mpos; br, bc = bpos
        dr, dc = mr-br, mc-bc
        is_4adj = abs(dr)+abs(dc)==1
        results = {}
        if is_4adj:
            for (r,c) in new_cells:
                if dr != 0:
                    if dr>0 and r<mr: continue
                    if dr<0 and r>mr: continue
                    nr, nc = mr+br-r, c
                else:
                    if dc>0 and c<mc: continue
                    if dc<0 and c>mc: continue
                    nr, nc = r, mc+bc-c
                if 0<=nr<R and 0<=nc<C and (nr,nc) not in new_cells:
                    results[(nr,nc)] = True
        else:
            mid_r, mid_c = (mr+br)/2, (mc+bc)/2
            dir_r, dir_c = mr-mid_r, mc-mid_c
            for (r,c) in new_cells:
                dot = (r-mid_r)*dir_r + (c-mid_c)*dir_c
                if dot <= 0: continue
                nr, nc = mr+br-r, mc+bc-c
                if 0<=nr<R and 0<=nc<C and (nr,nc) not in new_cells:
                    results[(nr,nc)] = True
        return results

    def find_best_s2(s1_clean, mpos, adj_all, pi, min_cells=2):
        best_s2 = set()
        best_s2_count = -1
        best_s2_dist = -1
        if not s1_clean:
            return best_s2
        s1_cr = sum(r for r,c in s1_clean)/len(s1_clean)
        s1_cc = sum(c for r,c in s1_clean)/len(s1_clean)
        for si, secondary in enumerate(adj_all):
            if si == pi: continue
            s2 = do_step2(s1_clean, mpos, secondary)
            s2_valid = {p for p in s2 if grid[p[0]][p[1]] == bg and p not in s1_clean}
            if len(s2_valid) < min_cells: continue
            s2_cr = sum(r for r,c in s2_valid)/max(len(s2_valid),1)
            s2_cc = sum(c for r,c in s2_valid)/max(len(s2_valid),1)
            dist = math.sqrt((s2_cr-s1_cr)**2+(s2_cc-s1_cc)**2)
            if len(s2_valid) > best_s2_count or (len(s2_valid)==best_s2_count and dist > best_s2_dist):
                best_s2_count = len(s2_valid)
                best_s2_dist = dist
                best_s2 = s2_valid
        return best_s2

    # --- Handle single-color components (symmetry completion) ---
    for root, cells in components.items():
        colors = Counter(non_bg[p] for p in cells)
        if len(colors) != 1:
            continue
        col = list(colors.keys())[0]
        cell_set = set(cells)
        # Find the row/col with most cells to determine axis
        row_counts = Counter(r for r,c in cells)
        col_counts = Counter(c for r,c in cells)
        max_row_count = max(row_counts.values())
        max_col_count = max(col_counts.values())
        
        if max_row_count >= max_col_count:
            # Vertical axis: find the row with most cells and its midline
            best_row = max(row_counts, key=row_counts.get)
            row_cells = [c for r,c in cells if r == best_row]
            axis = (min(row_cells) + max(row_cells)) / 2
            for (r,c) in list(cell_set):
                nc = int(2*axis - c)
                if 0<=nc<C and (r,nc) not in cell_set and grid[r][nc] == bg:
                    grid[r][nc] = col
                    cell_set.add((r,nc))
        else:
            # Horizontal axis
            best_col = max(col_counts, key=col_counts.get)
            col_cells = [r for r,c in cells if c == best_col]
            axis = (min(col_cells) + max(col_cells)) / 2
            for (r,c) in list(cell_set):
                nr = int(2*axis - r)
                if 0<=nr<R and (nr,c) not in cell_set and grid[nr][c] == bg:
                    grid[nr][c] = col
                    cell_set.add((nr,c))

    # --- Handle multi-color components ---
    for root, cells in components.items():
        colors = Counter(non_bg[p] for p in cells)
        if len(colors) < 2: continue
        body_color = colors.most_common(1)[0][0]
        body_cells = set(p for p in cells if non_bg[p] == body_color)
        markers = [(p, non_bg[p]) for p in cells if non_bg[p] != body_color]
        n_markers = len(markers)
        body_size = len(body_cells)

        # Special case: body_size=1 with 1 marker (2-cell objects)
        if body_size == 1 and n_markers == 1:
            mpos, mcol = markers[0]
            bpos = list(body_cells)[0]
            mr, mc = mpos
            br, bc = bpos
            dr, dc = mr-br, mc-bc  # direction from body to marker
            
            if abs(dr)+abs(dc) == 1:  # 4-adjacent
                # Determine perpendicular expansion direction
                if dr == 0:  # horizontal body-marker
                    # Try UP and DOWN
                    for sign in [-1, 1]:
                        b_cells = [(br+sign*i, bc-dc) for i in range(1,3)] + [(br+sign*i, bc) for i in range(1,3)]
                        m_cells = [(mr+sign*i, mc) for i in range(1,3)] + [(mr+sign*i, mc+dc) for i in range(1,3)]
                        all_new = b_cells + m_cells
                        if all(0<=nr<R and 0<=nc<C and grid[nr][nc]==bg for nr,nc in all_new):
                            for nr,nc in b_cells:
                                grid[nr][nc] = body_color
                            for nr,nc in m_cells:
                                grid[nr][nc] = mcol
                            break
                else:  # vertical body-marker
                    for sign in [-1, 1]:
                        b_cells = [(br, bc+sign*i) for i in range(1,3)] + [(br-dr, bc+sign*i) for i in range(1,3)]
                        m_cells = [(mr, mc+sign*i) for i in range(1,3)] + [(mr+dr, mc+sign*i) for i in range(1,3)]
                        all_new = b_cells + m_cells
                        if all(0<=nr<R and 0<=nc<C and grid[nr][nc]==bg for nr,nc in all_new):
                            for nr,nc in b_cells:
                                grid[nr][nc] = body_color
                            for nr,nc in m_cells:
                                grid[nr][nc] = mcol
                            break
            continue

        for mpos, mcol in markers:
            mr, mc = mpos
            adj_all = []
            for dr2 in [-1,0,1]:
                for dc2 in [-1,0,1]:
                    if dr2==0 and dc2==0: continue
                    nb = (mr+dr2, mc+dc2)
                    if nb in body_cells: adj_all.append(nb)
            if not adj_all: continue

            # Check for equal perpendicular sides
            has_equal_perp = False
            if n_markers == 1:
                for adj in adj_all:
                    if abs(mr-adj[0])+abs(mc-adj[1]) == 1:  # 4-adj
                        dr_adj = mr-adj[0]; dc_adj = mc-adj[1]
                        perp_counts = {}
                        for (r2,c2) in body_cells:
                            if dr_adj != 0:
                                side = c2 - adj[1]
                            else:
                                side = r2 - adj[0]
                            if side != 0:
                                key = 1 if side > 0 else -1
                                perp_counts[key] = perp_counts.get(key, 0) + 1
                        cp = perp_counts.get(1,0)
                        cn = perp_counts.get(-1,0)
                        if cp > 0 and cn > 0 and cp == cn:
                            has_equal_perp = True
                            break

            best = None
            best_score = -1

            if has_equal_perp and n_markers == 1:
                # Try both exclusion sides for equal perpendicular
                for adj in adj_all:
                    if abs(mr-adj[0])+abs(mc-adj[1]) != 1: continue
                    for excl_side in [-1, 1]:
                        s1 = do_reflect(body_cells, mpos, adj, force_exclude_side=excl_side)
                        s1_clean = {p for p in s1 if grid[p[0]][p[1]] == bg}
                        best_s2 = find_best_s2(s1_clean, mpos, adj_all, adj_all.index(adj), min_cells=1)
                        score = len(s1_clean) + len(best_s2)
                        if score > best_score:
                            best_score = score
                            best = (s1_clean, best_s2, mcol, body_color)
            else:
                for pi, primary in enumerate(adj_all):
                    s1 = do_reflect(body_cells, mpos, primary)
                    s1_clean = {p for p in s1 if grid[p[0]][p[1]] == bg}
                    best_s2 = set()
                    if n_markers == 1:
                        best_s2 = find_best_s2(s1_clean, mpos, adj_all, pi, min_cells=2)
                    if n_markers == 1:
                        score = len(s1_clean) + len(best_s2)
                    else:
                        score = len(s1_clean)
                    if score > best_score:
                        best_score = score
                        best = (s1_clean, best_s2, mcol, body_color)

            if best:
                s1, s2, mc_col, bc_col = best
                for p in s1:
                    if grid[p[0]][p[1]] == bg:
                        grid[p[0]][p[1]] = mc_col
                for p in s2:
                    if grid[p[0]][p[1]] == bg:
                        grid[p[0]][p[1]] = bc_col

    return grid
