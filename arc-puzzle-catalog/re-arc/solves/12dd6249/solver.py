from collections import Counter, deque

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    H = len(input_grid)
    W = len(input_grid[0])
    bg = Counter(v for row in input_grid for v in row).most_common(1)[0][0]
    out = [row[:] for row in input_grid]
    
    visited = set()
    regions = []
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] != bg and (r,c) not in visited:
                q = deque([(r,c)])
                visited.add((r,c))
                cells = [(r,c)]
                while q:
                    cr, cc = q.popleft()
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<H and 0<=nc<W and (nr,nc) not in visited and input_grid[nr][nc] != bg:
                            visited.add((nr,nc))
                            q.append((nr,nc))
                            cells.append((nr,nc))
                vals = Counter(input_grid[cr][cc] for cr,cc in cells)
                main = vals.most_common(1)[0][0]
                marker = None
                for cr,cc in cells:
                    if input_grid[cr][cc] != main:
                        marker = (cr, cc, input_grid[cr][cc])
                        break
                cell_set = set(cells)
                minr = min(cr for cr,cc in cells)
                maxr = max(cr for cr,cc in cells)
                minc = min(cc for cr,cc in cells)
                maxc = max(cc for cr,cc in cells)
                regions.append({
                    'main': main, 'cells': cell_set,
                    'r0': minr, 'c0': minc, 'r1': maxr, 'c1': maxc,
                    'H': maxr-minr+1, 'W': maxc-minc+1,
                    'marker': marker
                })
    
    marked = [r for r in regions if r['marker']]
    unmarked = [r for r in regions if not r['marker']]
    
    # Compute marked extensions info
    marked_info = []
    for reg in marked:
        mr, mc, mv = reg['marker']
        lr, lc = mr - reg['r0'], mc - reg['c0']
        d = None
        if lr == 0: d = 'UP'
        elif lr == reg['H']-1: d = 'DOWN'
        elif lc == 0: d = 'LEFT'
        elif lc == reg['W']-1: d = 'RIGHT'
        if d is None: continue
        
        if d in ('UP','DOWN'):
            opp = lr if d=='DOWN' else reg['H']-1-lr
            half = opp
            ext_start = reg['r1']+1 if d=='DOWN' else 0
            ext_end = H-1 if d=='DOWN' else reg['r0']-1
        else:
            opp = lc if d=='RIGHT' else reg['W']-1-lc
            half = opp
            ext_start = reg['c1']+1 if d=='RIGHT' else 0
            ext_end = W-1 if d=='LEFT' else reg['c0']-1
        
        marked_info.append({
            'reg': reg, 'dir': d, 'opp': opp,
            'ext_start': min(ext_start, ext_end),
            'ext_end': max(ext_start, ext_end),
            'span': 2*opp+1, 'mr': mr, 'mc': mc, 'mv': mv
        })
    
    # Apply marked extensions
    for mi in marked_info:
        reg = mi['reg']
        d = mi['dir']
        mr, mc, mv = mi['mr'], mi['mc'], mi['mv']
        opp = mi['opp']
        half = opp
        
        if d in ('UP', 'DOWN'):
            span_c0, span_c1 = mc - half, mc + half
            edge_row = [mv if c == mc else reg['main'] for c in range(span_c0, span_c1+1)]
            rng = range(reg['r1']+1, H) if d == 'DOWN' else range(reg['r0']-1, -1, -1)
            for r in rng:
                for i, c in enumerate(range(span_c0, span_c1+1)):
                    if 0 <= c < W and out[r][c] == bg:
                        out[r][c] = edge_row[i]
        else:
            span_r0, span_r1 = mr - half, mr + half
            edge_col = [mv if r == mr else reg['main'] for r in range(span_r0, span_r1+1)]
            rng = range(reg['c1']+1, W) if d == 'RIGHT' else range(reg['c0']-1, -1, -1)
            for c in rng:
                for i, r in enumerate(range(span_r0, span_r1+1)):
                    if 0 <= r < H and out[r][c] == bg:
                        out[r][c] = edge_col[i]
    
    # Process unmarked regions
    min_span = min((mi['span'] for mi in marked_info), default=0)
    
    for reg in unmarked:
        rH, rW = reg['H'], reg['W']
        
        # Direction: perpendicular to longest dim, toward nearest grid edge
        if rH >= rW:
            left_gap, right_gap = reg['c0'], W-1-reg['c1']
            if left_gap <= right_gap and left_gap > 0:
                best_dir, ext_len = 'LEFT', left_gap
            elif right_gap > 0:
                best_dir, ext_len = 'RIGHT', right_gap
            else: continue
        else:
            up_gap, down_gap = reg['r0'], H-1-reg['r1']
            if up_gap <= down_gap and up_gap > 0:
                best_dir, ext_len = 'UP', up_gap
            elif down_gap > 0:
                best_dir, ext_len = 'DOWN', down_gap
            else: continue
        
        span = min_span if min_span > 0 else max(rH, rW)
        
        # Determine start position
        if best_dir in ('LEFT', 'RIGHT'):
            # Find first row in block that's within any marked extension's row range
            start = reg['r0']
            for mi in marked_info:
                if mi['dir'] in ('UP', 'DOWN'):
                    # Marked extends vertically; find row intersection
                    inter_start = max(reg['r0'], mi['ext_start'])
                    inter_end = min(reg['r1'], mi['ext_end'])
                    if inter_start <= inter_end:
                        start = max(start, inter_start)
            
            # Also consider overlap with other unmarked blocks
            if len(unmarked) > 1:
                overlap_r0 = max(u['r0'] for u in unmarked)
                start = max(start, overlap_r0)
            
            span_r0 = start
            span_r1 = min(start + span - 1, reg['r1'])
            
            if best_dir == 'LEFT':
                for c in range(reg['c0']-1, reg['c0']-1-ext_len, -1):
                    for r in range(span_r0, span_r1+1):
                        if 0<=r<H and 0<=c<W and out[r][c] == bg:
                            out[r][c] = reg['main']
            else:
                for c in range(reg['c1']+1, reg['c1']+1+ext_len):
                    for r in range(span_r0, span_r1+1):
                        if 0<=r<H and 0<=c<W and out[r][c] == bg:
                            out[r][c] = reg['main']
        else:
            start = reg['c0']
            for mi in marked_info:
                if mi['dir'] in ('LEFT', 'RIGHT'):
                    inter_start = max(reg['c0'], mi['ext_start'])
                    inter_end = min(reg['c1'], mi['ext_end'])
                    if inter_start <= inter_end:
                        start = max(start, inter_start)
            if len(unmarked) > 1:
                overlap_c0 = max(u['c0'] for u in unmarked)
                start = max(start, overlap_c0)
            span_c0 = start
            span_c1 = min(start + span - 1, reg['c1'])
            if best_dir == 'UP':
                for r in range(reg['r0']-1, reg['r0']-1-ext_len, -1):
                    for c in range(span_c0, span_c1+1):
                        if 0<=r<H and 0<=c<W and out[r][c] == bg:
                            out[r][c] = reg['main']
            else:
                for r in range(reg['r1']+1, reg['r1']+1+ext_len):
                    for c in range(span_c0, span_c1+1):
                        if 0<=r<H and 0<=c<W and out[r][c] == bg:
                            out[r][c] = reg['main']
    
    return out
