from collections import Counter
from typing import List
Grid = List[List[int]]

def transform(grid_in: Grid) -> Grid:
    H, W = len(grid_in), len(grid_in[0])
    OH, OW = 3*H, 3*W
    bg = Counter(v for row in grid_in for v in row).most_common(1)[0][0]
    nonbg = {(r,c): grid_in[r][c] for r in range(H) for c in range(W) if grid_in[r][c] != bg}
    if not nonbg:
        return [[bg]*OW for _ in range(OH)]

    cc = Counter(nonbg.values())
    if len(cc) >= 2:
        main_color = cc.most_common(1)[0][0]
        marker_color = cc.most_common()[-1][0]
    else:
        marker_color = cc.most_common(1)[0][0]
        main_color = marker_color

    marker_cells = [(r,c) for (r,c),v in nonbg.items() if v == marker_color]

    # Try each marker position to find best chain
    best_chain = []
    best_mp = None
    best_groups_list = []

    for dr in range(2):
        for dc in range(2):
            groups = []
            for mr, mc in marker_cells:
                tr, tc = mr-dr, mc-dc
                if tr<0 or tc<0 or tr+1>=H or tc+1>=W: continue
                vals = [grid_in[tr+ddr][tc+ddc] for ddr in range(2) for ddc in range(2)]
                cnt = Counter(vals)
                if len(cnt)==2 and min(cnt.values())==1:
                    groups.append((tr,tc))
            groups = sorted(set(groups))
            if not groups: continue

            gs = set(groups)
            chains = []
            used = set()
            for gr,gc in groups:
                if (gr,gc) in used: continue
                ch = [(gr,gc)]; used.add((gr,gc))
                cr,cc2 = gr,gc
                while (cr+2,cc2-2) in gs and (cr+2,cc2-2) not in used:
                    cr,cc2=cr+2,cc2-2; ch.append((cr,cc2)); used.add((cr,cc2))
                cr,cc2 = gr,gc
                while (cr-2,cc2+2) in gs and (cr-2,cc2+2) not in used:
                    cr,cc2=cr-2,cc2+2; ch.insert(0,(cr,cc2)); used.add((cr,cc2))
                chains.append(ch)

            # Find active chain (longest, then extendable)
            active = None
            for ch in chains:
                if len(ch) > 1:
                    if active is None or len(ch) > len(active): active = ch
            if active is None:
                for ch in chains:
                    gr,gc = ch[0]
                    if (gr-2>=0 and gc+2<W) or (gr+3<H and gc-2>=0):
                        if active is None: active = ch; break

            if active is not None:
                if len(active) > len(best_chain):
                    best_chain = active
                    best_mp = (dr, dc)
                    best_groups_list = groups

    chain = best_chain
    if not chain:
        return [[bg]*OW for _ in range(OH)]

    # Determine the "odd cell" position within the group to get step
    # The odd cell is the one different from the other 3 in the 2x2 group
    gr0, gc0 = chain[0]
    vals = [(grid_in[gr0+ddr][gc0+ddc], (ddr,ddc)) for ddr in range(2) for ddc in range(2)]
    cnt = Counter(v for v,_ in vals)
    odd_color = min(cnt, key=cnt.get)
    odd_pos = next(pos for v,pos in vals if v == odd_color)
    step = 2 if odd_pos in [(1,0),(0,0)] else 4

    chain_cells_set: set = set()
    for gr,gc in chain:
        for ddr in range(2):
            for ddc in range(2):
                chain_cells_set.add((gr+ddr, gc+ddc))
    d_min = min(ir+ic for ir,ic in chain_cells_set)
    d_max = max(ir+ic for ir,ic in chain_cells_set)

    top_group = min(chain, key=lambda g:(g[0],-g[1]))
    bot_group = max(chain, key=lambda g:(g[0],-g[1]))
    base_L = 3*top_group[1] + step

    # Compute min_ic accounting for 1-step extension downward-left
    min_ic_chain = min(gc for _,gc in chain)
    ext_ir = bot_group[0] + 2
    ext_ic = bot_group[1] - 2
    if ext_ir + 1 < H and ext_ic >= 0:
        min_ic = ext_ic
    else:
        min_ic = min_ic_chain
    col_min_s = 3*min_ic + (2 if step==2 else 0)
    row_min_s = max(0, 3*top_group[0] - step)
    row_max_s = min(OH-1, row_min_s + base_L - col_min_s + 3*(d_max-d_min) + step//2 - 1)
    col_max_s = min(OW-1, base_L + 3)
    base_T = row_min_s + base_L - col_min_s

    def L(r: int) -> int:
        if r < row_min_s or r > row_max_s: return OW
        return max(col_min_s, base_L - step*((r - row_min_s)//step))

    def T(c: int) -> int:
        if c < col_min_s or c > col_max_s: return OH
        return max(row_min_s, base_T - step*((c - col_min_s)//step))

    def D_interior(c: int) -> int:
        base_d = 3*(d_max - d_min) + step//2
        if step == 2:
            h = (c - col_min_s) // step
            return base_d + 2*(h % 2)
        return base_d

    edge_ic_threshold = d_max - d_min + min_ic + 1

    def D_edge_val(ic: int) -> int:
        ir_lo = max(0, d_min + 1 - ic)
        ir_hi = min(H - 1, d_max - 1 - ic)
        num_int = max(0, ir_hi - ir_lo + 1)
        return 3 * num_int + 1

    B = [None]*OW
    for c in range(col_min_s, min(col_max_s+1, OW)):
        tc = T(c)
        if tc >= OH: continue
        ic = c // 3
        subc = c % 3
        if ic >= edge_ic_threshold and subc > 0:
            d = D_edge_val(ic)
        else:
            d = D_interior(c)
        B[c] = min(row_max_s, tc + d - 1)

    fill_color = main_color if main_color != bg else marker_color

    out = [[bg]*OW for _ in range(OH)]
    for r in range(row_min_s, row_max_s+1):
        lv = L(r)
        for c in range(lv, min(col_max_s+1, OW)):
            if B[c] is not None and r <= B[c]:
                out[r][c] = fill_color
    return out
