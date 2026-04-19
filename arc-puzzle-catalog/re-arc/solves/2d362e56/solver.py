from collections import Counter, deque
import math

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    vis = [[False]*cols for _ in range(rows)]
    comps = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not vis[r][c]:
                q = deque([(r, c)]); vis[r][c] = True; cells = []
                while q:
                    cr, cc = q.popleft(); cells.append((cr, cc, grid[cr][cc]))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not vis[nr][nc] and grid[nr][nc]!=bg:
                            vis[nr][nc] = True; q.append((nr, nc))
                comps.append(cells)

    templates = []; ntc = {}
    for comp in comps:
        cnt = Counter(c for _,_,c in comp); body = cnt.most_common(1)[0][0]
        if len(comp) >= 5 and cnt[body]/len(comp) >= 0.5:
            templates.append(comp)
        else:
            for r, c, col in comp: ntc[(r, c)] = col

    # Isolated pixels that fall inside a template's bounding box are part of that
    # template (disconnected by a bg gap but still semantically part of the shape).
    for template in templates:
        min_r = min(r for r,c,_ in template); max_r = max(r for r,c,_ in template)
        min_c = min(c for r,c,_ in template); max_c = max(c for r,c,_ in template)
        for key in list(ntc):
            r, c = key
            if min_r <= r <= max_r and min_c <= c <= max_c:
                template.append((r, c, ntc.pop(key)))

    tfns = [
        (lambda dr,dc:(dr,dc), 1),
        (lambda dr,dc:(dc,-dr), 1),
        (lambda dr,dc:(-dr,-dc), 1),
        (lambda dr,dc:(-dc,dr), 1),
        (lambda dr,dc:(dr,-dc), -1),
        (lambda dr,dc:(-dr,dc), -1),
        (lambda dr,dc:(dc,dr), -1),
        (lambda dr,dc:(-dc,-dr), -1),
    ]

    output = [[bg]*cols for _ in range(rows)]
    for template in templates:
        cnt = Counter(c for _,_,c in template); bc = cnt.most_common(1)[0][0]
        markers = [(r, c, col) for r, c, col in template if col != bc]
        if not markers: continue
        ref_r, ref_c, ref_col = markers[0]
        other_rel = [(r-ref_r, c-ref_c, col) for r, c, col in markers[1:]]
        all_rel = [(r-ref_r, c-ref_c, col) for r, c, col in template]
        ref_cands = [(r, c) for (r, c), col in ntc.items() if col == ref_col]
        used = set()
        for cand_r, cand_c in ref_cands:
            opts = []
            for ti, (tfn, det) in enumerate(tfns):
                match = True; mk = [(cand_r, cand_c)]
                for odr, odc, ocol in other_rel:
                    tdr, tdc = tfn(odr, odc); tr, tc = cand_r+tdr, cand_c+tdc
                    if ntc.get((tr, tc)) != ocol: match = False; break
                    mk.append((tr, tc))
                if not match: continue
                valid = True; bp = []
                for odr, odc, col in all_rel:
                    tdr, tdc = tfn(odr, odc); tr, tc = cand_r+tdr, cand_c+tdc
                    if not (0 <= tr < rows and 0 <= tc < cols): valid = False; break
                    if col == bc: bp.append((tr, tc))
                if not valid or not bp: continue
                pcr = sum(r for r,c in mk)/len(mk); pcc = sum(c for r,c in mk)/len(mk)
                bcr = sum(r for r,c in bp)/len(bp); bcc = sum(c for r,c in bp)/len(bp)
                dp = math.sqrt((bcr-pcr)**2 + (bcc-pcc)**2)
                # Tiebreaker: prefer higher ti except adiag (7) which loses to 90cw (1)
                key = 0 if ti == 7 else -ti
                # Position-aware correction for {90ccw(3),diag(6)} tie:
                # in top half, 90ccw (body above) beats diag (body below)
                if ti == 3 and cand_r < rows // 2 and bcr < pcr:
                    key -= 1000
                if ti == 6 and cand_r < rows // 2 and bcr > pcr:
                    key += 1000
                opts.append((dp, key, ti, tfn, mk))
            if opts:
                opts.sort(key=lambda x: (x[0], x[1]))
                _,_,_,best_tfn,mk = opts[0]
                pk = tuple(sorted(mk))
                if pk in used: continue
                used.add(pk)
                for odr, odc, col in all_rel:
                    tdr, tdc = best_tfn(odr, odc); tr, tc = cand_r+tdr, cand_c+tdc
                    if 0 <= tr < rows and 0 <= tc < cols: output[tr][tc] = col
    return output
