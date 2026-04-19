import json, numpy as np
from collections import Counter

with open('/tmp/rearc_agent_solves/5e4b6a73.json') as f:
    data = json.load(f)

# Import the solver module
import importlib.util
spec = importlib.util.spec_from_file_location('s', '/tmp/rearc_agent_solves/5e4b6a73_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

for ti, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    out_expected = np.array(ex['output'])
    H, W = inp.shape
    
    counts = Counter(inp.flatten())
    bg = counts.most_common(1)[0][0]
    
    # Find components
    visited = np.zeros_like(inp, dtype=bool)
    components = []
    for r in range(H):
        for c in range(W):
            if inp[r,c] != bg and not visited[r,c]:
                queue = [(r,c)]
                visited[r,c] = True
                cells = [(r,c)]
                idx = 0
                while idx < len(queue):
                    cr, cc = queue[idx]; idx += 1
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<H and 0<=nc<W and not visited[nr,nc] and inp[nr,nc] != bg:
                            visited[nr,nc] = True
                            queue.append((nr,nc))
                            cells.append((nr,nc))
                rows_c = [x[0] for x in cells]
                cols_c = [x[1] for x in cells]
                rmin, rmax = min(rows_c), max(rows_c)
                cmin, cmax = min(cols_c), max(cols_c)
                components.append({
                    'cells': cells, 'bbox': (rmin,cmin,rmax,cmax),
                    'area': (rmax-rmin+1)*(cmax-cmin+1), 'fill_area': len(cells)
                })
    
    components.sort(key=lambda x: x['area'], reverse=True)
    fill_colors = Counter()
    for r,c in components[0]['cells']:
        fill_colors[inp[r,c]] += 1
    fill = fill_colors.most_common(1)[0][0]
    
    rects = []
    marker = None
    for comp in components:
        fill_count = sum(1 for r,c in comp['cells'] if inp[r,c] == fill)
        if comp['area'] >= 20 and fill_count >= comp['fill_area'] * 0.5:
            rects.append(comp)
        else:
            if marker is None or comp['area'] < marker['area']:
                marker = comp
    
    mr1,mc1,mr2,mc2 = marker['bbox']
    mh, mw = mr2-mr1+1, mc2-mc1+1
    
    marker_pat = {}
    for r,c in marker['cells']:
        marker_pat[(r-mr1, c-mc1)] = int(inp[r,c])
    
    fill_r, fill_c = None, None
    for (r,c), v in marker_pat.items():
        if v == fill:
            fill_r, fill_c = r, c
            break
    if fill_r is None:
        fill_r, fill_c = mh//2, mw//2
    
    # Check for explicit anchors in rects
    rect_anchors = {}
    for ri, comp in enumerate(rects):
        rmin,cmin,rmax,cmax = comp['bbox']
        for r in range(rmin, rmax+1):
            for c in range(cmin, cmax+1):
                v = int(inp[r,c])
                if v != fill and v != bg:
                    rect_anchors[ri] = (r,c)
    
    # Detect trail
    trail_dirs = []
    left_cells = [(r,c,v) for (r,c),v in marker_pat.items() if r == fill_r and c < fill_c and v != fill and v != bg]
    if left_cells: trail_dirs.append(('left', left_cells[0][2]))
    right_cells = [(r,c,v) for (r,c),v in marker_pat.items() if r == fill_r and c > fill_c and v != fill and v != bg]
    if right_cells: trail_dirs.append(('right', right_cells[0][2]))
    down_bg = [(r,c) for (r,c),v in marker_pat.items() if c == fill_c and r > fill_r and v == bg]
    if down_bg: trail_dirs.append(('down_bg', bg))
    down_cells = [(r,c,v) for (r,c),v in marker_pat.items() if c == fill_c and r > fill_r and v != fill and v != bg]
    if down_cells: trail_dirs.append(('down', down_cells[0][2]))
    
    trail_axis = None
    for td, _ in trail_dirs:
        if td.startswith('down') or td.startswith('up'):
            trail_axis = 'row'
            break
        elif td.startswith('left') or td.startswith('right'):
            trail_axis = 'col'
            break
    if trail_axis is None: trail_axis = 'row'
    
    mcr = (mr1+mr2)/2.0
    mcc = (mc1+mc2)/2.0
    
    print(f"\n{'='*60}")
    print(f"TRAIN {ti}: bg={bg}, fill={fill}")
    print(f"Marker bbox=({mr1},{mc1},{mr2},{mc2}), fill_pos=({fill_r},{fill_c})")
    print(f"Marker center=({mcr},{mcc})")
    print(f"Trail dirs: {trail_dirs}, trail_axis={trail_axis}")
    print(f"Marker pattern: {marker_pat}")
    
    for ri, comp in enumerate(rects):
        rmin,cmin,rmax,cmax = comp['bbox']
        if ri in rect_anchors:
            ar, ac = rect_anchors[ri]
            method = "explicit"
        else:
            if trail_axis == 'row':
                ar = mod.compute_coord_trail(mcr, mr1, mr2, fill_r, rmin, rmax)
                ac = mod.compute_coord_nontrail(mcc, mc1, mc2, fill_c, cmin, cmax)
            else:
                ar = mod.compute_coord_nontrail(mcr, mr1, mr2, fill_r, rmin, rmax)
                ac = mod.compute_coord_trail(mcc, mc1, mc2, fill_c, cmin, cmax)
            method = "computed"
        
        # Find expected anchor from output
        exp_anchor = None
        for r in range(rmin+1, rmax):
            for c in range(cmin+1, cmax):
                if out_expected[r,c] == fill:
                    non_fill = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)] if out_expected[r+dr,c+dc] != fill)
                    if non_fill >= 3:
                        exp_anchor = (r,c)
                        break
            if exp_anchor: break
        if not exp_anchor:
            for r in range(rmin+1, rmax):
                for c in range(cmin+1, cmax):
                    if out_expected[r,c] == fill:
                        non_fill = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)] if out_expected[r+dr,c+dc] != fill)
                        if non_fill >= 2:
                            exp_anchor = (r,c)
                            break
                if exp_anchor: break
        
        print(f"\n  Rect {ri} ({rmin},{cmin},{rmax},{cmax}):")
        print(f"    Computed anchor: ({ar},{ac}) [{method}]")
        print(f"    Expected anchor: {exp_anchor}")
        if exp_anchor and (ar, ac) != exp_anchor:
            print(f"    *** MISMATCH ***")

