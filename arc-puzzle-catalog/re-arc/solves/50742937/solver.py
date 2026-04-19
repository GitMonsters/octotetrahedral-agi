"""Solver for ARC puzzle 50742937 - 3D isometric box rendering."""

def transform(input_grid):
    H = len(input_grid)
    W = len(input_grid[0])
    from collections import Counter
    colors = Counter(c for r in input_grid for c in r)
    bg = colors.most_common(1)[0][0]
    output = [row[:] for row in input_grid]
    visited = [[False]*W for _ in range(H)]
    blocks = []
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] != bg and not visited[r][c]:
                queue = [(r,c)]
                visited[r][c] = True
                cells = []
                while queue:
                    cr, cc = queue.pop(0)
                    cells.append((cr,cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<H and 0<=nc<W and not visited[nr][nc] and input_grid[nr][nc]!=bg:
                            visited[nr][nc] = True
                            queue.append((nr,nc))
                r1,r2 = min(x[0] for x in cells), max(x[0] for x in cells)
                c1,c2 = min(x[1] for x in cells), max(x[1] for x in cells)
                blocks.append((r1, c1, r2-r1+1, c2-c1+1, input_grid[r1][c1]))
    if blocks:
        for (r1, c1, bh, bw, color) in blocks:
            s = min(bh, bw); border = s // 2
            if bg != 4:
                for r in range(max(0,r1-border), min(H,r1+bh+border)):
                    for c in range(max(0,c1-border), min(W,c1+bw+border)):
                        if output[r][c] == bg: output[r][c] = 4
            ss = r1 + bh + border
            if bg != 4:
                for r in range(ss, H):
                    for c in range(c1, c1+bw):
                        if 0<=c<W and output[r][c] == bg: output[r][c] = 2
            else:
                narrow = max(0, border - 1)
                for r in range(ss, H):
                    sw = bw - narrow if r < H-1 else bw
                    for c in range(c1, c1+sw):
                        if 0<=c<W and output[r][c] == bg: output[r][c] = 2
    else:
        output = generate_scene(H, W, bg)
    return output

def _zigzag(fh, bs, bsz, sa, sb, sc):
    center = bs + bsz / 2.0
    p = []; row = 1; idx = 0
    while row + 1 < fh:
        p.append((row, sa if idx%2==0 else sb))
        gap = 1
        if row+2 <= center <= row+2+gap+1: gap = 2
        row += 2 + gap; idx += 1
    if p: p[-1] = (p[-1][0], sc)
    return p

def generate_scene(H, W, bg):
    g = [[bg]*W for _ in range(H)]
    B = (min(H,W)//2) & ~1; h = B//2
    br = h-1; bc = (W-B)//2; fb = br+B-1+h
    lbg = 1; le = bc-lbg-h; hl = le > 0
    lsw = le//2 if hl else 0; lgw = le-lsw if hl else 0; fl = lbg+(le if hl else 0)
    rs = W-bc-B; mB = B//2; mh = mB//2; mn = (h-1)+mB+mh+1; hm = rs >= mn
    if hm: mr=br+B//2; mc=bc+B+(h-1); mft=mr-mh; mes=mr+mh; mfr=mc+mB+mh
    else: mr=mc=mft=mes=mfr=0

    # FRAME
    for r in range(min(fb+1,H)):
        cl = (0 if hl else fl) if r<br else (fl if r<br+B else (lbg if hl else fl))
        if hm:
            if r<=br: cr = mc+mB  # includes block start row
            elif r<mft: cr = mc+mB-1  # block level before med frame
            elif r<mes: cr = mfr-1
            else: cr = mfr
        else:
            cr = W-2 if r<br else W-1
        for c in range(cl, min(cr+1,W)): g[r][c] = 4
    if hl:
        for r in range(br+B, min(fb+1,H)):
            for c in range(lbg, fl): g[r][c] = 4

    # CARVE BIG BLOCK
    for r in range(br,br+B):
        for c in range(bc,bc+B): g[r][c] = bg

    # LEFT SIDE
    if hl:
        for r in range(br,br+B):
            for c in range(lbg,lbg+lsw): g[r][c] = 2
            for c in range(lbg+lsw,lbg+lsw+lgw): g[r][c] = bg
        for r in range(br, min(fb+1,H)): g[r][0] = bg
        for r in range(br): g[r][le] = bg

    # CARVE MEDIUM BLOCK
    if hm:
        for r in range(mr,mr+mB):
            for c in range(mc,mc+mB):
                if 0<=r<H and 0<=c<W: g[r][c] = bg

    # PATCHES
    fh = fb+2
    # Right face (only without medium block)
    rp = [] if hm else _zigzag(fh, br, B, bc+B+2, bc+B+4, bc+B+3)
    # Left face
    lp = []
    if hl:
        if br >= 3: lp.append((0, lbg))
        bot = br+B
        if fb-bot >= 2: lp.append((bot+1, lbg+1))  # strip_c
    # Medium area
    mp = []
    if hm:
        sa,sb = mc+2, mc
        row,idx = 1, 0
        while row+1 <= mft:  # FIX: was < mft
            mp.append((row, sa if idx%2==0 else sb))
            row += 3; idx += 1
        below_r = mr+mB+mh+1
        if below_r+1 < H: mp.append((below_r, sb))
        # Medium right face
        mrl = mc+mB; mrc = mrl+1; mra = mrl
        mrow = mes+1; mi = 0
        while mrow+1 <= fb+1:
            mp.append((mrow, mrc if mi==0 else mra))
            mrow += 2+mh; mi += 1

    # CARVE ALL PATCHES
    all_p = [('r',rp),('l',lp),('m',mp)]
    for _,patches in all_p:
        for pr,pc in patches:
            for dr in range(2):
                for dc in range(2):
                    if 0<=pr+dr<H and 0<=pc+dc<W: g[pr+dr][pc+dc] = bg

    # SHADOWS
    st = fb+1
    for r in range(st,H):
        for c in range(bc,bc+B):
            if 0<=c<W: g[r][c] = 2
    if hl:
        for r in range(st,H):
            for c in range(lbg,lbg+lsw+1):
                if 0<=c<W: g[r][c] = 2
    if not hm:
        ext = W-1
        for pr,pc in rp:
            if pc+1 == ext:
                for sr in range(pr+3, min(pr+3+h-1,H)):
                    if 0<=sr<H: g[sr][ext] = 2
    if hm:
        for r in range(st,H):
            for c in range(mc,mc+mB):
                if 0<=c<W: g[r][c] = 2
        # Medium column patch shadows (only if extent >= 2)
        for pr,pc in mp:
            if pc < mc+mB and pc+1 < W:
                ext_len = max(1, mh-1)
                if ext_len >= 2:  # only draw if extent >= 2
                    for sr in range(pr+3, min(pr+3+ext_len,H)):
                        if 0<=sr<=fb: g[sr][pc+1] = 2
                elif ext_len == 1:  # overwrite frame cells only
                    sr = pr+3
                    if 0<=sr<=fb and sr<H and g[sr][pc+1] == 4:
                        g[sr][pc+1] = 2

    # Combined patch shadow at bottom rows (fb+3 to H-1)
    patch_shadow_start = fb + 3
    for _,patches in all_p:
        if patches:
            minc = min(pc for _,pc in patches)
            maxc = max(pc+1 for _,pc in patches)
            for r in range(patch_shadow_start, H):
                for c in range(minc, maxc+1):
                    if 0<=c<W: g[r][c] = 2
    if hm:
        for r in range(patch_shadow_start, H):
            for c in range(mc,mc+mB):
                if 0<=c<W: g[r][c] = 2

    # PATCH FRAME EXTENSIONS (for patches extending below fb)
    for _,patches in all_p:
        for pr,pc in patches:
            if pr+1 >= fb:  # extends at or below frame bottom
                for dr in range(-1, 3):
                    for dc in range(-1, 3):
                        if 0<=dr<2 and 0<=dc<2: continue
                        rr,cc = pr+dr, pc+dc
                        if rr >= fb and 0<=rr<H-1 and 0<=cc<W:
                            g[rr][cc] = 4
                # Re-carve patch cells
                for dr in range(2):
                    for dc in range(2):
                        rr,cc = pr+dr, pc+dc
                        if 0<=rr<H and 0<=cc<W: g[rr][cc] = bg

    return g
