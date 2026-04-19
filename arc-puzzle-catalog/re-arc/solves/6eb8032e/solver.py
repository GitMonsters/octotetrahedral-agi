from collections import Counter
from itertools import combinations, product

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    bg = Counter(sum(grid, [])).most_common(1)[0][0]
    
    colors = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                colors.setdefault(grid[r][c], []).append((r, c))
    
    if not colors:
        return [row[:] for row in grid]
    
    num_colors = len(colors)
    step = 4
    all_3x3 = set((r,c) for r in range(3) for c in range(3))
    
    def find_origins(pts, color, template, ref=None):
        pts_set = set(pts)
        cands = {}
        for r, c in pts:
            for dr, dc in template:
                org = (r-dr, c-dc)
                if org in cands: continue
                conflict = False
                covers = set()
                for tdr, tdc in template:
                    pr, pc = org[0]+tdr, org[1]+tdc
                    if 0 <= pr < rows and 0 <= pc < cols:
                        v = grid[pr][pc]
                        if v != bg and v != color:
                            conflict = True; break
                        if (pr,pc) in pts_set:
                            covers.add((pr,pc))
                if not conflict and covers:
                    cands[org] = covers
        
        if ref is not None:
            cands = {o:c for o,c in cands.items()
                    if (o[0]-ref[0])%step==0 and (o[1]-ref[1])%step==0}
        
        cand_list = sorted(cands.items(), key=lambda x: -len(x[1]))
        best = [None]
        def bt(rem, idx, chosen):
            if not rem:
                if best[0] is None or len(chosen) < len(best[0]):
                    best[0] = chosen[:]
                return
            if idx >= len(cand_list): return
            if best[0] and len(chosen) >= len(best[0]): return
            org, cov = cand_list[idx]
            inter = rem & cov
            if inter: bt(rem - inter, idx+1, chosen + [org])
            bt(rem, idx+1, chosen)
        bt(pts_set, 0, [])
        return best[0] or []
    
    def find_ref(template):
        all_cands = {}
        for color, pts in colors.items():
            for r,c in pts:
                for dr,dc in template:
                    org = (r-dr, c-dc)
                    all_cands.setdefault(org, set()).add((r,c))
        
        keys = sorted(all_cands.keys(), key=lambda k: -len(all_cands[k]))
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                o1, o2 = keys[i], keys[j]
                if ((o1[0]-o2[0])%step==0 and (o1[1]-o2[1])%step==0
                    and len(all_cands[o1])>=2 and len(all_cands[o2])>=2
                    and not (all_cands[o1] & all_cands[o2])):
                    return o1
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                o1, o2 = keys[i], keys[j]
                if ((o1[0]-o2[0])%step==0 and (o1[1]-o2[1])%step==0
                    and not (all_cands[o1] & all_cands[o2])):
                    return o1
        return None
    
    def expand_and_validate(initial_known):
        """Expand known offsets, then validate: keep only offsets seen at grid-aligned origins."""
        known = set(initial_known)
        for iteration in range(10):
            template = sorted(known)
            if not template: break
            ref = find_ref(template)
            if ref is None: break
            
            expanded = set(known)
            for color, pts in colors.items():
                pts_set = set(pts)
                for r, c in pts:
                    for a_dr in range(3):
                        for a_dc in range(3):
                            org = (r-a_dr, c-a_dc)
                            if (org[0]-ref[0])%step != 0 or (org[1]-ref[1])%step != 0:
                                continue
                            matches = 0
                            conflict = False
                            for tdr, tdc in template:
                                pr, pc = org[0]+tdr, org[1]+tdc
                                if 0<=pr<rows and 0<=pc<cols:
                                    v = grid[pr][pc]
                                    if v != bg and v != color:
                                        conflict = True; break
                                    if (pr,pc) in pts_set: matches += 1
                            if conflict or matches < 1: continue
                            for r2, c2 in pts:
                                dr, dc = r2-org[0], c2-org[1]
                                if 0<=dr<3 and 0<=dc<3:
                                    expanded.add((dr,dc))
            
            if expanded == known: break
            known = expanded
        
        # Validate: keep only offsets that appear at grid-aligned origins with ≥2 matches
        if not known: return known
        template = sorted(known)
        ref = find_ref(template)
        if ref is None: return known
        
        validated = set()
        for color, pts in colors.items():
            pts_set = set(pts)
            for r, c in pts:
                for a_dr in range(3):
                    for a_dc in range(3):
                        org = (r-a_dr, c-a_dc)
                        if (org[0]-ref[0])%step != 0 or (org[1]-ref[1])%step != 0:
                            continue
                        # Count matches with expanded template
                        matches = set()
                        conflict = False
                        for tdr, tdc in template:
                            pr, pc = org[0]+tdr, org[1]+tdc
                            if 0<=pr<rows and 0<=pc<cols:
                                v = grid[pr][pc]
                                if v != bg and v != color:
                                    conflict = True; break
                                if (pr,pc) in pts_set:
                                    matches.add((tdr,tdc))
                        if conflict or len(matches) < 2: continue
                        validated.update(matches)
        
        return validated if validated else known
    
    def get_clusters(pts):
        clusters = set()
        for pr, pc in pts:
            for a_dr in range(3):
                for a_dc in range(3):
                    org = (pr-a_dr, pc-a_dc)
                    offs = frozenset(
                        (r2-org[0], c2-org[1])
                        for r2, c2 in pts
                        if 0<=r2-org[0]<3 and 0<=c2-org[1]<3
                    )
                    if len(offs) >= 2:
                        clusters.add(offs)
        return [set(c) for c in clusters]
    
    # Check for complete template (≥5 pixel cluster)
    complete_template = None
    complete_origin = None
    complete_color = None
    for color, pts in colors.items():
        pts_set = set(pts)
        for pr, pc in pts:
            for a_dr in range(3):
                for a_dc in range(3):
                    org = (pr-a_dr, pc-a_dc)
                    offs = set()
                    for r2,c2 in pts:
                        dr,dc = r2-org[0], c2-org[1]
                        if 0<=dr<3 and 0<=dc<3:
                            offs.add((dr,dc))
                    if len(offs) >= 5 and (complete_template is None or len(offs) > len(complete_template)):
                        complete_template = sorted(offs)
                        complete_origin = org
                        complete_color = color
    
    if complete_template:
        template = complete_template
        ref = complete_origin
        origins_map = {}
        origins_map[complete_color] = [complete_origin]
        ok = True
        for color in colors:
            if color != complete_color:
                orgs = find_origins(colors[color], color, template, ref)
                if not orgs: ok = False; break
                origins_map[color] = orgs
        if not ok: origins_map = None
    else:
        # Try different initial cluster combinations
        clusters_per_color = {c: get_clusters(pts) for c, pts in colors.items()}
        color_list = list(colors.keys())
        
        best_known = set()
        combos = list(product(*(clusters_per_color[c] for c in color_list)))
        if len(combos) > 200:
            combos = combos[:200]
        
        for combo in combos:
            initial = set()
            for cl in combo:
                initial.update(cl)
            expanded = expand_and_validate(initial)
            if len(expanded) > len(best_known):
                best_known = expanded
        
        known = best_known
        unknown = sorted(all_3x3 - known)
        
        # Gap inference
        origins_map = None
        template = None
        
        if 2 <= len(unknown) <= 5:
            template_known = sorted(known)
            ref = find_ref(template_known)
            
            origins_tmp = {}
            for color, pts in colors.items():
                origins_tmp[color] = find_origins(pts, color, template_known, ref)
            
            offset_counts = Counter()
            for color, orgs in origins_tmp.items():
                for org in orgs:
                    for r, c in colors[color]:
                        dr, dc = r-org[0], c-org[1]
                        if 0<=dr<3 and 0<=dc<3:
                            offset_counts[(dr,dc)] += 1
            
            unique = {o for o, cnt in offset_counts.items() if cnt == 1}
            
            candidates = set()
            for uo in unique:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = uo[0]+dr, uo[1]+dc
                    if 0<=nr<3 and 0<=nc<3 and (nr,nc) not in known:
                        candidates.add((nr,nc))
            
            if not candidates:
                candidates = set(unknown)
            
            def adj8(pos):
                return sum(1 for kr,kc in known
                         if abs(pos[0]-kr)<=1 and abs(pos[1]-kc)<=1 and (kr,kc)!=pos)
            
            scored = sorted(candidates, key=lambda p: (adj8(p), -p[0], -p[1]))
            n_gaps = min(2, len(scored))
            gaps = set(scored[:n_gaps])
            template = sorted(all_3x3 - gaps)
        elif len(unknown) < 2:
            template = sorted(known) if len(known) >= 5 else sorted(all_3x3)
        
        if template:
            ref = find_ref(template)
            origins_map = {}
            ok = True
            for color, pts in colors.items():
                orgs = find_origins(pts, color, template, ref)
                if not orgs: ok = False; break
                origins_map[color] = orgs
            
            if ok:
                for color, pts in colors.items():
                    covered = set()
                    for org in origins_map[color]:
                        for dr,dc in template:
                            if (org[0]+dr, org[1]+dc) in set(pts):
                                covered.add((org[0]+dr, org[1]+dc))
                    if covered != set(pts): ok = False; break
            
            if not ok: origins_map = None
        
        # Fallback: brute force
        if origins_map is None:
            for n_gaps in [2, 1, 3, 4, 0]:
                for gaps in combinations(sorted(all_3x3), n_gaps):
                    t = sorted(all_3x3 - set(gaps))
                    if len(t) < 3: continue
                    ref = find_ref(t)
                    if ref is None: continue
                    om = {}
                    valid = True
                    for color, pts in colors.items():
                        orgs = find_origins(pts, color, t, ref)
                        if not orgs: valid = False; break
                        om[color] = orgs
                    if not valid: continue
                    for color, pts in colors.items():
                        covered = set()
                        for org in om[color]:
                            for dr,dc in t:
                                if (org[0]+dr, org[1]+dc) in set(pts):
                                    covered.add((org[0]+dr, org[1]+dc))
                        if covered != set(pts): valid = False; break
                    if valid:
                        template = t; origins_map = om; break
                if origins_map: break
    
    if origins_map is None:
        return [row[:] for row in grid]
    
    # Chain extension
    output = [[bg]*cols for _ in range(rows)]
    
    for color, orgs in origins_map.items():
        all_orgs = set(orgs)
        for org in orgs:
            if color == complete_color: continue
            
            if complete_color is not None:
                ref_pt = complete_origin
            elif len(orgs) >= 2:
                others = [o for o in orgs if o != org]
                ref_pt = (sum(o[0] for o in others)/len(others),
                         sum(o[1] for o in others)/len(others))
            else:
                other_all = [o for c2,os2 in origins_map.items() if c2!=color for o in os2]
                if other_all:
                    ref_pt = (sum(o[0] for o in other_all)/len(other_all),
                             sum(o[1] for o in other_all)/len(other_all))
                else: continue
            
            drs = (1 if org[0]>ref_pt[0]+.01 else (-1 if org[0]<ref_pt[0]-.01 else 0))
            dcs = (1 if org[1]>ref_pt[1]+.01 else (-1 if org[1]<ref_pt[1]-.01 else 0))
            
            if num_colors == 1:
                if drs == 0: drs = -1 if org[0] <= rows-1-org[0] else 1
                if dcs == 0: dcs = -1 if org[1] <= cols-1-org[1] else 1
            
            if drs == 0 and dcs == 0: continue
            
            cs = (drs*step, dcs*step)
            k = 1
            while True:
                no = (org[0]+k*cs[0], org[1]+k*cs[1])
                if not any(0<=no[0]+dr<rows and 0<=no[1]+dc<cols for dr,dc in template): break
                all_orgs.add(no); k += 1
        
        for org in all_orgs:
            for dr,dc in template:
                r,c = org[0]+dr, org[1]+dc
                if 0<=r<rows and 0<=c<cols:
                    output[r][c] = color
    
    return output
