"""ARC Puzzle 720006c5 - Template stamping with adjacency validation."""
import numpy as np
from collections import deque, Counter
from itertools import product


def transform(input_grid):
    grid = np.array(input_grid)
    H, W = grid.shape
    bg = Counter(grid.flatten().tolist()).most_common(1)[0][0]
    
    ISOS = [
        lambda r,c:(r,c), lambda r,c:(c,-r), lambda r,c:(-r,-c), lambda r,c:(-c,r),
        lambda r,c:(r,-c), lambda r,c:(-r,c), lambda r,c:(c,r), lambda r,c:(-c,-r),
    ]
    def iso(r,c,i): return ISOS[i](r,c)
    
    def find_comps():
        vis=set(); comps=[]
        for r in range(H):
            for c in range(W):
                if grid[r,c]!=bg and (r,c) not in vis:
                    comp={}; q=deque([(r,c)]); vis.add((r,c))
                    while q:
                        cr,cc=q.popleft(); comp[(cr,cc)]=int(grid[cr,cc])
                        for dr in [-1,0,1]:
                            for dc in [-1,0,1]:
                                if dr==0 and dc==0: continue
                                nr,nc=cr+dr,cc+dc
                                if 0<=nr<H and 0<=nc<W and (nr,nc) not in vis and grid[nr,nc]!=bg:
                                    vis.add((nr,nc)); q.append((nr,nc))
                    comps.append(comp)
        return comps
    
    def find_holes(comp):
        cells=set(comp.keys())
        mnr=min(r for r,c in cells)-1; mxr=max(r for r,c in cells)+1
        mnc=min(c for r,c in cells)-1; mxc=max(c for r,c in cells)+1
        reach=set(); q=deque()
        for r in range(max(0,mnr),min(H,mxr+1)):
            for c in range(max(0,mnc),min(W,mxc+1)):
                if (r==mnr or r==mxr or c==mnc or c==mxc) and (r,c) not in cells and 0<=r<H and 0<=c<W:
                    reach.add((r,c)); q.append((r,c))
        while q:
            cr,cc=q.popleft()
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=cr+dr,cc+dc
                if mnr<=nr<=mxr and mnc<=nc<=mxc and 0<=nr<H and 0<=nc<W and (nr,nc) not in cells and (nr,nc) not in reach:
                    reach.add((nr,nc)); q.append((nr,nc))
        return {(r,c) for r in range(max(0,mnr+1),min(H,mxr)) for c in range(max(0,mnc+1),min(W,mxc)) if (r,c) not in cells and (r,c) not in reach and grid[r,c]==bg}
    
    def norm(comp):
        cells=list(comp.items())
        mr=min(r for (r,c),v in cells); mc=min(c for (r,c),v in cells)
        return frozenset(((r-mr,c-mc),v) for (r,c),v in cells)
    
    def all_isos_fn(shape):
        res=set()
        for i in range(8):
            t=[(iso(r,c,i),v) for (r,c),v in shape]
            mr=min(r for (r,c),v in t); mc=min(c for (r,c),v in t)
            res.add(frozenset(((r-mr,c-mc),v) for (r,c),v in t))
        return res
    
    def stamp_cells(sm1, iso_id, all_off):
        cells = set()
        for (dr,dc),v in all_off:
            tr,tc=iso(dr,dc,iso_id)
            cells.add((sm1[0]+tr, sm1[1]+tc))
        return cells
    
    def stamps_adjacent(cells_list):
        """Check if any two stamp cell sets are 8-adjacent."""
        for i in range(len(cells_list)):
            expanded = set()
            for r,c in cells_list[i]:
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        expanded.add((r+dr,c+dc))
            for j in range(i+1, len(cells_list)):
                if expanded & cells_list[j]:
                    return True
        return False
    
    comps = find_comps()
    aug = []
    for comp in comps:
        holes=find_holes(comp)
        a=dict(comp)
        for h in holes: a[h]=bg
        aug.append(a)
    
    templates=[]; non_t=[]
    for comp,a in zip(comps,aug):
        if len(set(a.values()))>=2 and len(a)>3:
            templates.append(a)
        else: non_t.append(comp)
    
    if not templates:
        return _no_template(comps,grid,bg,H,W,norm,all_isos_fn)
    
    templates.sort(key=len, reverse=False)  # smallest first
    ext={}
    for comp in non_t: ext.update(comp)
    
    out=[[bg]*W for _ in range(H)]
    used=set()
    
    for tmpl in templates:
        colors=Counter(tmpl.values())
        mk_col=min(colors, key=colors.get)
        bd_col=max(colors, key=colors.get)
        mk_pos=sorted([(r,c) for (r,c),v in tmpl.items() if v==mk_col])
        bd_pos=sorted([(r,c) for (r,c),v in tmpl.items() if v==bd_col])
        n_mk=len(mk_pos)
        ref=mk_pos[0] if n_mk>=1 else bd_pos[0]
        all_off=[((r-ref[0],c-ref[1]),v) for (r,c),v in tmpl.items()]
        
        ext_mk=sorted([(r,c) for (r,c),v in ext.items() if v==mk_col and (r,c) not in used])
        ext_bd_set=set((r,c) for (r,c),v in ext.items() if v==bd_col and (r,c) not in used)
        
        if n_mk>=2:
            m1,m2=mk_pos[0],mk_pos[1]
            mv=(m2[0]-m1[0],m2[1]-m1[1])
            bd_off=[(b[0]-m1[0],b[1]-m1[1]) for b in bd_pos]
            
            best_stamps=None; best_score=None
            for a_off in bd_off:
                pair_cands={}
                for i in range(len(ext_mk)):
                    for j in range(i+1,len(ext_mk)):
                        s1,s2=ext_mk[i],ext_mk[j]
                        key=tuple(sorted([s1,s2]))
                        for sm1,sm2 in [(s1,s2),(s2,s1)]:
                            sv=(sm2[0]-sm1[0],sm2[1]-sm1[1])
                            for iso_id in range(8):
                                if iso(mv[0],mv[1],iso_id)!=sv: continue
                                ta=iso(a_off[0],a_off[1],iso_id)
                                eb=(sm1[0]+ta[0],sm1[1]+ta[1])
                                if eb not in ext_bd_set: continue
                                ok=True
                                for (dr,dc),v in all_off:
                                    tr,tc=iso(dr,dc,iso_id)
                                    nr,nc=sm1[0]+tr,sm1[1]+tc
                                    if not(0<=nr<H and 0<=nc<W): ok=False; break
                                if not ok: continue
                                pair_cands.setdefault(key,[]).append((sm1,sm2,eb,iso_id))
                
                pairs=sorted(pair_cands.keys())
                if not pairs: continue
                if any(len(pair_cands.get(p,[]))==0 for p in pairs): continue
                
                sizes=[len(pair_cands[p]) for p in pairs]
                total=1
                for s in sizes: total*=s
                
                if total <= 4096:
                    ranges=[range(len(pair_cands[p])) for p in pairs]
                    for combo in product(*ranges):
                        stamps=[]; u=set(used); valid=True
                        for pi, ci in enumerate(combo):
                            cand=pair_cands[pairs[pi]][ci]
                            sm1_,sm2_,eb_,iso_id_=cand
                            if sm1_ in u or sm2_ in u or eb_ in u:
                                valid=False; break
                            stamps.append(cand); u|={sm1_,sm2_,eb_}
                        if not valid or len(stamps)!=len(pairs): continue
                        
                        # Check no two stamps are 8-adjacent
                        cells_list = [stamp_cells(c[0],c[3],all_off) for c in stamps]
                        if stamps_adjacent(cells_list): continue
                        
                        # Score: prefer balanced rotation/reflection count
                        n_ref = sum(1 for c in stamps if c[3] >= 4)
                        target = len(stamps) / 2.0
                        score = (abs(n_ref - target), n_ref)  # prefer closer to half, then fewer reflections
                        
                        if best_stamps is None or len(stamps)>len(best_stamps) or (len(stamps)==len(best_stamps) and (best_score is None or score < best_score)):
                            best_stamps=stamps; best_score=score
                else:
                    sp=sorted(range(len(pairs)), key=lambda i: sizes[i])
                    def solve(idx,u):
                        if idx==len(sp): return []
                        pi=sp[idx]
                        for cand in pair_cands[pairs[pi]]:
                            sm1_,sm2_,eb_,iso_id_=cand
                            if sm1_ in u or sm2_ in u or eb_ in u: continue
                            rest=solve(idx+1,u|{sm1_,sm2_,eb_})
                            if rest is not None: return [cand]+rest
                        return None
                    result=solve(0,set(used))
                    if result and (best_stamps is None or len(result)>len(best_stamps)):
                        best_stamps=result
            
            if best_stamps:
                for sm1,sm2,eb,iso_id in best_stamps:
                    for (dr,dc),v in all_off:
                        tr,tc=iso(dr,dc,iso_id)
                        nr,nc=sm1[0]+tr,sm1[1]+tc
                        if 0<=nr<H and 0<=nc<W: out[nr][nc]=v
                    used.update([sm1,sm2,eb])
        
        elif n_mk==1:
            m1=mk_pos[0]
            bd_off=[(b[0]-m1[0],b[1]-m1[1]) for b in bd_pos]
            best_stamps=None
            for a_off in bd_off:
                mk_cands={}
                for sm in ext_mk:
                    for iso_id in range(8):
                        ta=iso(a_off[0],a_off[1],iso_id)
                        eb=(sm[0]+ta[0],sm[1]+ta[1])
                        if eb not in ext_bd_set: continue
                        ok=True
                        for (dr,dc),v in all_off:
                            tr,tc=iso(dr,dc,iso_id)
                            nr,nc=sm[0]+tr,sm[1]+tc
                            if not(0<=nr<H and 0<=nc<W): ok=False; break
                        if not ok: continue
                        mk_cands.setdefault(sm,[]).append((sm,eb,iso_id))
                
                mks=sorted(mk_cands.keys())
                if not mks: continue
                if any(len(mk_cands.get(m,[]))==0 for m in mks): continue
                
                sizes=[len(mk_cands[m]) for m in mks]
                total=1
                for s in sizes: total*=s
                
                if total<=4096:
                    ranges=[range(len(mk_cands[m])) for m in mks]
                    for combo in product(*ranges):
                        stamps=[]; u=set(used); valid=True
                        for mi,ci in enumerate(combo):
                            cand=mk_cands[mks[mi]][ci]
                            sm_,eb_,iso_id_=cand
                            if sm_ in u or eb_ in u: valid=False; break
                            stamps.append(cand); u|={sm_,eb_}
                        if not valid or len(stamps)!=len(mks): continue
                        
                        cells_list=[stamp_cells(c[0],c[2],all_off) for c in stamps]
                        if stamps_adjacent(cells_list): continue
                        
                        if best_stamps is None or len(stamps)>len(best_stamps):
                            best_stamps=stamps
                            break
                else:
                    sp=sorted(range(len(mks)), key=lambda i: sizes[i])
                    def solve1(idx,u):
                        if idx==len(sp): return []
                        mi=sp[idx]
                        for cand in mk_cands[mks[mi]]:
                            sm_,eb_,iso_id_=cand
                            if sm_ in u or eb_ in u: continue
                            rest=solve1(idx+1,u|{sm_,eb_})
                            if rest is not None: return [cand]+rest
                        return None
                    result=solve1(0,set(used))
                    if result and (best_stamps is None or len(result)>len(best_stamps)):
                        best_stamps=result
            
            if best_stamps:
                for sm,eb,iso_id in best_stamps:
                    for (dr,dc),v in all_off:
                        tr,tc=iso(dr,dc,iso_id)
                        nr,nc=sm[0]+tr,sm[1]+tc
                        if 0<=nr<H and 0<=nc<W: out[nr][nc]=v
                    used.update([sm,eb])
    
    return out


def _no_template(comps,grid,bg,H,W,norm,all_isos_fn):
    if not comps: return [list(row) for row in grid]
    exact_groups = {}
    for i,comp in enumerate(comps):
        ns=norm(comp)
        exact_groups.setdefault(ns,[]).append(i)
    dup_indices = set()
    for ns, indices in exact_groups.items():
        if len(indices) > 1:
            for idx in indices: dup_indices.add(idx)
    if not dup_indices: return [list(row) for row in grid]
    centers = {}
    for i, comp in enumerate(comps):
        cells = list(comp.keys())
        centers[i] = (sum(r for r,c in cells)/len(cells), sum(c for r,c in cells)/len(cells))
    min_dist = float('inf'); min_pair = None
    for i in range(len(comps)):
        for j in range(i+1, len(comps)):
            ci, cj = centers[i], centers[j]
            d = ((ci[0]-cj[0])**2 + (ci[1]-cj[1])**2)**0.5
            if d < min_dist: min_dist=d; min_pair=(i,j)
    if min_pair is None: return [list(row) for row in grid]
    i, j = min_pair
    to_remove = None
    if i in dup_indices and j not in dup_indices: to_remove = i
    elif j in dup_indices and i not in dup_indices: to_remove = j
    elif i in dup_indices and j in dup_indices:
        to_remove = max(i, j, key=lambda k: min(comps[k].keys()))
    out = [list(row) for row in grid]
    if to_remove is not None:
        for r,c in comps[to_remove]: out[r][c] = bg
    return out
