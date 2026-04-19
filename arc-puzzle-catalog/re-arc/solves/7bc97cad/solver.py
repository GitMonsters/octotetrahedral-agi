import numpy as np
from collections import Counter, defaultdict

def transform(grid_list):
    grid = np.array(grid_list, dtype=int)
    H, W = grid.shape
    out = grid.copy()
    
    bg = Counter(grid.flatten()).most_common(1)[0][0]
    
    visited = np.zeros_like(grid, dtype=bool)
    components = []
    
    def flood_fill(r, c):
        stack = [(r, c)]
        cells = []
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= H or cc < 0 or cc >= W: continue
            if visited[cr, cc] or grid[cr, cc] == bg: continue
            visited[cr, cc] = True
            cells.append((cr, cc, int(grid[cr, cc])))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc+dc))
        return cells
    
    for r in range(H):
        for c in range(W):
            if not visited[r, c] and grid[r, c] != bg:
                comp = flood_fill(r, c)
                if comp:
                    components.append(comp)
    
    def normalize(comp):
        min_r = min(r for r,c,v in comp)
        min_c = min(c for r,c,v in comp)
        return frozenset((r-min_r, c-min_c, v) for r,c,v in comp)
    
    def is_shape_subset(smaller, larger):
        for tr, tc, tv in larger:
            for sr, sc, sv in smaller:
                if sv == tv:
                    or_r, or_c = sr - tr, sc - tc
                    if all((pr-or_r, pc-or_c, pv) in larger for pr,pc,pv in smaller):
                        return True
        return False
    
    def is_4connected(cells):
        if not cells: return True
        pos_set = set((r,c) for r,c,v in cells)
        start = next(iter(pos_set))
        vis = {start}
        stk = [start]
        while stk:
            r, c = stk.pop()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (nr, nc) in pos_set and (nr, nc) not in vis:
                    vis.add((nr, nc)); stk.append((nr, nc))
        return len(vis) == len(pos_set)
    
    def try_match(comp, template):
        comp_pos = set((r,c) for r,c,v in comp)
        results = []
        tried = set()
        for tr, tc, tv in template:
            for cr, cc, cv in comp:
                if cv == tv:
                    or_r, or_c = cr - tr, cc - tc
                    if (or_r, or_c) in tried: continue
                    tried.add((or_r, or_c))
                    if not all((pr-or_r, pc-or_c, pv) in template for pr,pc,pv in comp):
                        continue
                    missing, valid, overlap = [], True, 0
                    for sr, sc, sv in template:
                        gr, gc = or_r + sr, or_c + sc
                        if (gr, gc) in comp_pos: continue
                        if 0 <= gr < H and 0 <= gc < W:
                            if grid[gr, gc] == bg:
                                missing.append((gr, gc, sv))
                            elif grid[gr, gc] == sv:
                                overlap += 1
                            else:
                                valid = False; break
                        else:
                            valid = False; break
                    if valid and missing:
                        results.append((or_r, or_c, missing, overlap))
        return results
    
    multi_shapes = defaultdict(list)
    single_cells = []
    for i, comp in enumerate(components):
        if len(comp) > 1:
            multi_shapes[normalize(comp)].append(i)
        else:
            single_cells.append(i)
    
    all_multi = sorted(multi_shapes.keys(), key=lambda s: -len(s))
    
    # Template combination for small multi-color templates
    combined_templates = []
    mc_shapes = [s for s in all_multi if len(set(v for _,_,v in s)) > 1]
    sc_shapes = [s for s in all_multi if len(set(v for _,_,v in s)) == 1]
    
    for mc in mc_shapes:
        if len(mc) > 3: continue
        mc_colors = set(v for _,_,v in mc)
        for sc in sc_shapes:
            sc_color = next(iter(set(v for _,_,v in sc)))
            if sc_color not in mc_colors: continue
            mc_pos = set((r,c) for r,c,v in mc)
            best_combined, best_mr = None, -1
            for mc_r,mc_c in [(r,c) for r,c,v in mc if v==sc_color]:
                for sc_r,sc_c,_ in [(r,c,v) for r,c,v in sc if v==sc_color]:
                    for dr,dc in [(1,0),(0,1),(-1,0),(0,-1)]:
                        sr,sc2 = mc_r+dr-sc_r, mc_c+dc-sc_c
                        shifted = [(r+sr,c+sc2,v) for r,c,v in sc]
                        sp = set((r,c) for r,c,v in shifted)
                        if sp & mc_pos: continue
                        comb = list(mc) + list(shifted)
                        if not is_4connected(comb): continue
                        mr2 = max(r for r,c,v in shifted)
                        if best_combined is None or mr2 > best_mr:
                            best_combined = normalize(comb); best_mr = mr2
            if best_combined and len(best_combined) > len(mc):
                combined_templates.append(best_combined)
    
    templates = list(combined_templates)
    for shape in all_multi:
        is_sub = False
        for larger in templates:
            if len(larger) <= len(shape): continue
            if is_shape_subset(shape, larger):
                is_sub = True; break
        if not is_sub:
            templates.append(shape)
    
    # Phase 1: Multi-cell partial matches
    processed = set()
    for ci, comp in enumerate(components):
        if len(comp) <= 1: continue
        if normalize(comp) in templates:
            processed.add(ci); continue
        for tmpl in templates:
            if len(tmpl) <= len(comp): continue
            matches = try_match(comp, tmpl)
            if matches:
                matches.sort(key=lambda m: -m[3])
                for r,c,v in matches[0][2]: out[r,c] = v
                processed.add(ci); break
    
    # Phase 2-3: Single cells
    sc_data = [(components[i][0][0], components[i][0][1], components[i][0][2], i) 
               for i in single_cells]
    sc_map = {(r,c): (v,i) for r,c,v,i in sc_data}
    
    ad_colors, pair_top = set(), {}
    for r,c,v,i in sc_data:
        if (r+1,c-1) in sc_map and sc_map[(r+1,c-1)][0] != v:
            nv = sc_map[(r+1,c-1)][0]
            ad_colors.update([v, nv]); pair_top[v] = nv
    pair_bottom = {v:k for k,v in pair_top.items()}
    
    for r,c,v,ci in sc_data:
        if v in ad_colors:
            already = ((r+1,c-1) in sc_map and sc_map[(r+1,c-1)][0]!=v) or \
                      ((r-1,c+1) in sc_map and sc_map[(r-1,c+1)][0]!=v)
            if not already:
                if v in pair_top and 0<=r+1<H and 0<=c-1<W and out[r+1,c-1]==bg:
                    out[r+1,c-1] = pair_top[v]
                elif v in pair_bottom and 0<=r-1<H and 0<=c+1<W and out[r-1,c+1]==bg:
                    out[r-1,c+1] = pair_bottom[v]
            processed.add(ci)
        else:
            for tmpl in templates:
                if v not in set(tv for _,_,tv in tmpl): continue
                matches = try_match(components[ci], tmpl)
                if not matches: continue
                best = max(matches, key=lambda m: (m[3], -(r-m[0])))
                for mr,mc,mv in best[2]: out[mr,mc] = mv
                processed.add(ci); break
    
    # Phase 4: Markers
    remaining = [(r,c,v) for ci,comp in enumerate(components) 
                 if ci not in processed for r,c,v in comp]
    if not remaining: return out.tolist()
    
    rem_colors = set(v for _,_,v in remaining)
    if len(rem_colors) != 1 or not templates: return out.tolist()
    
    marker_color = list(rem_colors)[0]
    primary = templates[0]
    
    existing_tls = set()
    for ci,comp in enumerate(components):
        if normalize(comp) == primary:
            existing_tls.add((min(r for r,c,v in comp), min(c for r,c,v in comp)))
    for ci in processed:
        comp = components[ci]
        if len(comp) > 1 and normalize(comp) != primary:
            matches = try_match(comp, primary)
            if matches: existing_tls.add((matches[0][0], matches[0][1]))
    
    marker_pos = [(r,c) for r,c,v in remaining]
    
    # Find offset
    best_off, best_hits, best_mag = None, 0, float('inf')
    for dr in range(-20, 20):
        for dc in range(-20, 20):
            if dr==0 and dc==0: continue
            hits = sum(1 for mr,mc in marker_pos if (mr+dr,mc+dc) in existing_tls)
            mag = abs(dr)+abs(dc)
            if hits > best_hits or (hits == best_hits and mag < best_mag):
                best_hits, best_off, best_mag = hits, (dr,dc), mag
    
    if not best_off or best_hits < 1: return out.tolist()
    off_dr, off_dc = best_off
    
    # Claim
    claimed = set()
    for tl in existing_tls:
        exp = (tl[0]-off_dr, tl[1]-off_dc)
        for m in marker_pos:
            if m not in claimed and m == exp: claimed.add(m); break
    for tl in existing_tls:
        exp = (tl[0]-off_dr, tl[1]-off_dc)
        if exp in claimed: continue
        avail = [m for m in marker_pos if m not in claimed]
        if avail:
            best_m = min(avail, key=lambda m: abs(m[0]-exp[0])+abs(m[1]-exp[1]))
            claimed.add(best_m)
    
    # Assign markers to EXISTING blocks only
    m2b = {}
    for mr,mc in marker_pos:
        m2b[(mr,mc)] = min(existing_tls, key=lambda bl: abs(mr-bl[0])+abs(mc-bl[1]))
    
    bm = defaultdict(list)
    for (mr,mc), bl in m2b.items():
        bm[bl].append((mr-bl[0], mc-bl[1]))
    
    # Reference block = the one with most markers
    ref_bl = max(bm.items(), key=lambda x: len(x[1]))
    ref_tl, ref_offs = ref_bl
    ref_offs_set = set(ref_offs)
    
    # Generate new blocks
    for mr,mc in marker_pos:
        if (mr,mc) in claimed: continue
        tl_r, tl_c = mr+off_dr, mc+off_dc
        fits = True
        for sr,sc,sv in primary:
            gr, gc = tl_r+sr, tl_c+sc
            if not (0<=gr<H and 0<=gc<W) or (grid[gr,gc]!=bg and grid[gr,gc]!=sv):
                fits = False; break
        if fits:
            for sr,sc,sv in primary:
                gr, gc = tl_r+sr, tl_c+sc
                if out[gr,gc] == bg: out[gr,gc] = sv
    
    # Phase 5: Emission - only for blocks whose markers MATCH the reference pattern
    pl = list(primary)
    p_min_r, p_max_r = min(r for r,c,v in pl), max(r for r,c,v in pl)
    p_min_c, p_max_c = min(c for r,c,v in pl), max(c for r,c,v in pl)
    block_h = p_max_r - p_min_r + 1
    p_pos = set((r,c) for r,c,v in pl)
    missing_corners = [(r,c) for r in range(p_min_r, p_max_r+1) 
                       for c in range(p_min_c, p_max_c+1) if (r,c) not in p_pos]
    
    if missing_corners and len(ref_offs) >= 3:
        mc_r, mc_c = missing_corners[0]
        
        for bl_r, bl_c in existing_tls:
            bl_offs = set(bm.get((bl_r, bl_c), []))
            # Check: does this block's markers match the reference offsets?
            match_count = sum(1 for off in ref_offs if off in bl_offs)
            if match_count >= 3 and match_count == len(ref_offs):
                emit_positions = [
                    (bl_r + mc_r + block_h + 2, bl_c + mc_c - block_h),
                    (bl_r + mc_r + block_h + 2, bl_c + mc_c + 2 * block_h),
                ]
                for er, ec in emit_positions:
                    if 0 <= er < H and 0 <= ec < W and out[er, ec] == bg:
                        out[er, ec] = marker_color
    
    return out.tolist()
