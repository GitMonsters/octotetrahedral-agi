#!/usr/bin/env python3
"""
MEGA ARC SOLVER: combines ALL solver strategies into one runner.
Runs every strategy on every task, reports combined score.
"""

import os, sys, json, time, multiprocessing
import numpy as np
from collections import Counter, defaultdict
from scipy import ndimage


def grids_match(a, b):
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and np.array_equal(a, b)

def find_bg(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


# ═══════════════════════════════════════════════
# Neighborhood LUT
# ═══════════════════════════════════════════════

def get_nb(grid, r, c, radius=1):
    h, w = grid.shape
    vals = []
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            nr, nc = r+dr, c+dc
            vals.append(int(grid[nr, nc]) if 0 <= nr < h and 0 <= nc < w else -1)
    return tuple(vals)


def solve_nb_lut(task, radius=1):
    train = task['train']
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    lut = {}
    for ex in train:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        h, w = inp.shape
        for r in range(h):
            for c in range(w):
                key = get_nb(inp, r, c, radius)
                val = int(out[r, c])
                if key in lut and lut[key] != val:
                    return None
                lut[key] = val
    
    test_inp = np.array(task['test'][0]['input'])
    h, w = test_inp.shape
    result = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            key = get_nb(test_inp, r, c, radius)
            if key not in lut:
                return None
            result[r, c] = lut[key]
    
    return result.tolist(), f'nb_lut_r{radius}'


# ═══════════════════════════════════════════════
# Abstract Features (expanded)
# ═══════════════════════════════════════════════

def get_features(grid, r, c):
    h, w = grid.shape
    val = int(grid[r, c])
    bg = find_bg(grid)
    
    n4 = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < h and 0 <= nc < w:
            n4.append(int(grid[nr, nc]))
    n8 = list(n4)
    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < h and 0 <= nc < w:
            n8.append(int(grid[nr, nc]))
    
    nn_bg = sum(1 for n in n8 if n != bg)
    nn4_bg = sum(1 for n in n4 if n != bg)
    adj_same = sum(1 for n in n4 if n == val)
    adj_colors = tuple(sorted(set(n4)))
    
    feats = [
        ('v', val),
        ('v_nn', val, nn_bg),
        ('v_nc', val, adj_colors),
        ('v_border', val, r==0 or r==h-1 or c==0 or c==w-1),
        ('v_adj_same', val, adj_same),
        ('v_maj', val, Counter(n8).most_common(1)[0][0] if n8 else bg),
        ('v_parity', val, r%2, c%2),
        ('v_bg_adj', val, val==bg, nn4_bg),
        # New features
        ('v_row', val, r),
        ('v_col', val, c),
        ('v_rc', val, r, c),
        ('v_nn4', val, nn4_bg),
        ('v_rmod3', val, r%3, c%3),
        ('v_diag', val, (r+c)%2),
        ('v_diag2', val, (r-c)%3 if r>=c else (c-r)%3),
        ('v_dist_edge', val, min(r, c, h-1-r, w-1-c)),
        ('v_rmod', val, r % max(1, h//2)),
        ('v_cmod', val, c % max(1, w//2)),
        ('n4_sorted', tuple(sorted(n4))),
        ('n4_val', val, tuple(sorted(n4))),
        ('v_adj_diff', val, tuple(sorted(set(n4) - {val}))),
    ]
    return feats


def solve_abstract(task):
    train = task['train']
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    n_feats = len(get_features(np.array(train[0]['input']), 0, 0))
    
    for fi in range(n_feats):
        lut = {}
        ok = True
        for ex in train:
            inp, out = np.array(ex['input']), np.array(ex['output'])
            h, w = inp.shape
            for r in range(h):
                for c in range(w):
                    key = get_features(inp, r, c)[fi]
                    val = int(out[r, c])
                    if key in lut and lut[key] != val:
                        ok = False
                        break
                    lut[key] = val
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            test_inp = np.array(task['test'][0]['input'])
            h, w = test_inp.shape
            result = np.zeros((h, w), dtype=int)
            complete = True
            for r in range(h):
                for c in range(w):
                    key = get_features(test_inp, r, c)[fi]
                    if key not in lut:
                        complete = False
                        break
                    result[r, c] = lut[key]
                if not complete:
                    break
            if complete:
                return result.tolist(), f'abstract_f{fi}'
    
    return None


# ═══════════════════════════════════════════════
# Grid Partition (separator detection)
# ═══════════════════════════════════════════════

def find_seps(grid, bg=0):
    h, w = grid.shape
    srows = [(r, int(grid[r,0])) for r in range(h) 
             if len(set(grid[r].flatten().tolist())) == 1 and int(grid[r,0]) != bg]
    scols = [(c, int(grid[0,c])) for c in range(w) 
             if len(set(grid[:,c].flatten().tolist())) == 1 and int(grid[0,c]) != bg]
    return srows, scols


def get_subgrids(grid, srows, scols):
    h, w = grid.shape
    rows = [-1] + [r for r, _ in srows] + [h]
    cols = [-1] + [c for c, _ in scols] + [w]
    sgs = []
    for i in range(len(rows)-1):
        for j in range(len(cols)-1):
            r1, r2 = rows[i]+1, rows[i+1]
            c1, c2 = cols[j]+1, cols[j+1]
            if r1 < r2 and c1 < c2:
                sgs.append(((r1, c1, r2, c2), grid[r1:r2, c1:c2].copy()))
    return sgs


def solve_partition(task):
    train = task['train']
    inp0 = np.array(train[0]['input'])
    out0 = np.array(train[0]['output'])
    bg = find_bg(inp0)
    
    sr, sc = find_seps(inp0, bg)
    if not sr and not sc:
        return None
    
    sgs0 = get_subgrids(inp0, sr, sc)
    if not sgs0:
        return None
    
    # Try: output = specific subgrid by index
    for idx in range(len(sgs0)):
        if grids_match(sgs0[idx][1], out0):
            ok = True
            for ex in train[1:]:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                si, sci = find_seps(inp, find_bg(inp))
                sgs = get_subgrids(inp, si, sci)
                if idx >= len(sgs) or not grids_match(sgs[idx][1], out):
                    ok = False
                    break
            if ok:
                t = np.array(task['test'][0]['input'])
                st, sct = find_seps(t, find_bg(t))
                sgst = get_subgrids(t, st, sct)
                if idx < len(sgst):
                    return sgst[idx][1].tolist(), f'partition_idx{idx}'
    
    # Try: output = subgrid with most/least non-bg cells
    props = [
        ('most_nonbg', lambda sg, bg: np.count_nonzero(sg != bg), True),
        ('least_nonbg', lambda sg, bg: np.count_nonzero(sg != bg), False),
        ('most_colors', lambda sg, bg: len(set(sg.flatten().tolist()) - {bg}), True),
        ('least_colors', lambda sg, bg: len(set(sg.flatten().tolist()) - {bg}), False),
        ('most_unique', lambda sg, bg: len(set(sg.flatten().tolist())), True),
    ]
    
    for pname, pfn, want_max in props:
        # Find which subgrid matches output
        match_idx = None
        for i, (_, sg) in enumerate(sgs0):
            if grids_match(sg, out0):
                match_idx = i
                break
        if match_idx is None:
            continue
        
        # Is it the max/min?
        vals = [(pfn(sg, bg), i) for i, (_, sg) in enumerate(sgs0)]
        vals.sort(reverse=want_max)
        if vals[0][1] != match_idx:
            continue
        
        ok = True
        for ex in train[1:]:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            bgi = find_bg(inp)
            si, sci = find_seps(inp, bgi)
            sgs = get_subgrids(inp, si, sci)
            if not sgs:
                ok = False
                break
            v = [(pfn(sg, bgi), j) for j, (_, sg) in enumerate(sgs)]
            v.sort(reverse=want_max)
            if not grids_match(sgs[v[0][1]][1], out):
                ok = False
                break
        
        if ok:
            t = np.array(task['test'][0]['input'])
            bgt = find_bg(t)
            st, sct = find_seps(t, bgt)
            sgst = get_subgrids(t, st, sct)
            if sgst:
                v = [(pfn(sg, bgt), j) for j, (_, sg) in enumerate(sgst)]
                v.sort(reverse=want_max)
                return sgst[v[0][1]][1].tolist(), f'partition:{pname}'
    
    # Try: output = overlay/OR/AND of all subgrids
    if len(sgs0) >= 2:
        sg_shapes = [sg.shape for _, sg in sgs0]
        if len(set(sg_shapes)) == 1:
            sh = sg_shapes[0]
            
            # OR: non-bg in any → keep
            def overlay_or(sgs, bg):
                result = np.full(sh, bg, dtype=int)
                for _, sg in sgs:
                    mask = sg != bg
                    result[mask] = sg[mask]
                return result
            
            # AND: non-bg in all
            def overlay_and(sgs, bg):
                result = np.full(sh, bg, dtype=int)
                for r in range(sh[0]):
                    for c in range(sh[1]):
                        vals = [int(sg[r,c]) for _, sg in sgs if sg[r,c] != bg]
                        if len(vals) == len(sgs):
                            result[r, c] = Counter(vals).most_common(1)[0][0]
                return result
            
            # XOR: non-bg in exactly one
            def overlay_xor(sgs, bg):
                result = np.full(sh, bg, dtype=int)
                for r in range(sh[0]):
                    for c in range(sh[1]):
                        vals = [(i, int(sg[r,c])) for i, (_, sg) in enumerate(sgs) if sg[r,c] != bg]
                        if len(vals) == 1:
                            result[r, c] = vals[0][1]
                return result
            
            # Majority
            def overlay_majority(sgs, bg):
                result = np.full(sh, bg, dtype=int)
                for r in range(sh[0]):
                    for c in range(sh[1]):
                        vals = [int(sg[r,c]) for _, sg in sgs if sg[r,c] != bg]
                        if vals:
                            result[r, c] = Counter(vals).most_common(1)[0][0]
                return result
            
            for op_name, op_fn in [('or', overlay_or), ('and', overlay_and), 
                                    ('xor', overlay_xor), ('majority', overlay_majority)]:
                pred0 = op_fn(sgs0, bg)
                if grids_match(pred0, out0):
                    ok = True
                    for ex in train[1:]:
                        inp = np.array(ex['input'])
                        out = np.array(ex['output'])
                        bgi = find_bg(inp)
                        si, sci = find_seps(inp, bgi)
                        sgs = get_subgrids(inp, si, sci)
                        sgi_shapes = [sg.shape for _, sg in sgs]
                        if len(set(sgi_shapes)) != 1:
                            ok = False
                            break
                        pred = op_fn(sgs, bgi)
                        if not grids_match(pred, out):
                            ok = False
                            break
                    
                    if ok:
                        t = np.array(task['test'][0]['input'])
                        bgt = find_bg(t)
                        st, sct = find_seps(t, bgt)
                        sgst = get_subgrids(t, st, sct)
                        if sgst:
                            return op_fn(sgst, bgt).tolist(), f'partition_{op_name}'
    
    return None


# ═══════════════════════════════════════════════
# Color Map
# ═══════════════════════════════════════════════

def solve_colormap(task):
    train = task['train']
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    cmap = {}
    for ex in train:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        for iv, ov in zip(inp.flatten(), out.flatten()):
            iv, ov = int(iv), int(ov)
            if iv in cmap and cmap[iv] != ov:
                return None
            cmap[iv] = ov
    
    test_inp = np.array(task['test'][0]['input'])
    for v in test_inp.flatten():
        if int(v) not in cmap:
            return None
    pred = np.vectorize(lambda x: cmap[int(x)])(test_inp)
    return pred.tolist(), 'colormap'


# ═══════════════════════════════════════════════
# Geometric transforms
# ═══════════════════════════════════════════════

def solve_geometric(task):
    train = task['train']
    
    transforms = [
        ('fliplr', lambda g: np.fliplr(g)),
        ('flipud', lambda g: np.flipud(g)),
        ('rot90', lambda g: np.rot90(g, 1)),
        ('rot180', lambda g: np.rot90(g, 2)),
        ('rot270', lambda g: np.rot90(g, 3)),
        ('transpose', lambda g: g.T),
    ]
    
    for name, fn in transforms:
        ok = True
        for ex in train:
            if not grids_match(fn(np.array(ex['input'])), np.array(ex['output'])):
                ok = False
                break
        if ok:
            return fn(np.array(task['test'][0]['input'])).tolist(), name
    
    # Depth-2 compositions
    for n1, f1 in transforms:
        for n2, f2 in transforms:
            ok = True
            for ex in train:
                pred = f2(f1(np.array(ex['input'])))
                if not grids_match(pred, np.array(ex['output'])):
                    ok = False
                    break
            if ok:
                return f2(f1(np.array(task['test'][0]['input']))).tolist(), f'{n1}+{n2}'
    
    return None


# ═══════════════════════════════════════════════
# Scale operations
# ═══════════════════════════════════════════════

def solve_scale(task):
    train = task['train']
    ex0 = train[0]
    inp, out = np.array(ex0['input']), np.array(ex0['output'])
    ih, iw = inp.shape
    oh, ow = out.shape
    
    # Scale up
    for f in range(2, 6):
        if oh == ih * f and ow == iw * f:
            pred = np.repeat(np.repeat(inp, f, axis=0), f, axis=1)
            if grids_match(pred, out):
                ok = True
                for ex in train[1:]:
                    i, o = np.array(ex['input']), np.array(ex['output'])
                    p = np.repeat(np.repeat(i, f, axis=0), f, axis=1)
                    if not grids_match(p, o):
                        ok = False
                        break
                if ok:
                    t = np.array(task['test'][0]['input'])
                    return np.repeat(np.repeat(t, f, axis=0), f, axis=1).tolist(), f'scale_up_{f}'
    
    # Scale down
    for f in range(2, 6):
        if ih == oh * f and iw == ow * f:
            bg = find_bg(inp)
            pred = np.zeros((oh, ow), dtype=int)
            for r in range(oh):
                for c in range(ow):
                    block = inp[r*f:(r+1)*f, c*f:(c+1)*f]
                    vals = block.flatten()
                    nb = vals[vals != bg]
                    pred[r,c] = Counter(nb.tolist()).most_common(1)[0][0] if len(nb) > 0 else bg
            
            if grids_match(pred, out):
                ok = True
                for ex in train[1:]:
                    i, o = np.array(ex['input']), np.array(ex['output'])
                    bgi = find_bg(i)
                    p = np.zeros(o.shape, dtype=int)
                    for r in range(o.shape[0]):
                        for c in range(o.shape[1]):
                            block = i[r*f:(r+1)*f, c*f:(c+1)*f]
                            vals = block.flatten()
                            nb = vals[vals != bgi]
                            p[r,c] = Counter(nb.tolist()).most_common(1)[0][0] if len(nb) > 0 else bgi
                    if not grids_match(p, o):
                        ok = False
                        break
                if ok:
                    t = np.array(task['test'][0]['input'])
                    bgt = find_bg(t)
                    p = np.zeros((t.shape[0]//f, t.shape[1]//f), dtype=int)
                    for r in range(p.shape[0]):
                        for c in range(p.shape[1]):
                            block = t[r*f:(r+1)*f, c*f:(c+1)*f]
                            vals = block.flatten()
                            nb = vals[vals != bgt]
                            p[r,c] = Counter(nb.tolist()).most_common(1)[0][0] if len(nb) > 0 else bgt
                    return p.tolist(), f'scale_down_{f}'
    
    return None


# ═══════════════════════════════════════════════
# Tile
# ═══════════════════════════════════════════════

def solve_tile(task):
    train = task['train']
    ex0 = train[0]
    inp, out = np.array(ex0['input']), np.array(ex0['output'])
    ih, iw = inp.shape
    oh, ow = out.shape
    
    if oh < ih or ow < iw:
        return None
    if oh % ih != 0 or ow % iw != 0:
        return None
    
    rh, rw = oh // ih, ow // iw
    if rh == 1 and rw == 1:
        return None
    
    pred = np.tile(inp, (rh, rw))
    if not grids_match(pred, out):
        return None
    
    for ex in train[1:]:
        i, o = np.array(ex['input']), np.array(ex['output'])
        ih2, iw2 = i.shape
        oh2, ow2 = o.shape
        if oh2 % ih2 != 0 or ow2 % iw2 != 0:
            return None
        if oh2 // ih2 != rh or ow2 // iw2 != rw:
            return None
        if not grids_match(np.tile(i, (rh, rw)), o):
            return None
    
    t = np.array(task['test'][0]['input'])
    return np.tile(t, (rh, rw)).tolist(), f'tile_{rh}x{rw}'


# ═══════════════════════════════════════════════
# Crop
# ═══════════════════════════════════════════════

def solve_crop(task):
    train = task['train']
    
    # Crop to non-bg bbox
    bg = find_bg(np.array(train[0]['input']))
    
    ok = True
    for ex in train:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        rows, cols = np.where(inp != bg)
        if len(rows) == 0:
            ok = False
            break
        crop = inp[rows.min():rows.max()+1, cols.min():cols.max()+1]
        if not grids_match(crop, out):
            ok = False
            break
    
    if ok:
        t = np.array(task['test'][0]['input'])
        bgt = find_bg(t)
        rows, cols = np.where(t != bgt)
        if len(rows) > 0:
            return t[rows.min():rows.max()+1, cols.min():cols.max()+1].tolist(), 'crop_nonbg'
    
    # Crop to specific color's bbox
    inp0 = np.array(train[0]['input'])
    out0 = np.array(train[0]['output'])
    colors = set(inp0.flatten().tolist()) - {bg}
    
    for color in colors:
        rows, cols = np.where(inp0 == color)
        if len(rows) == 0:
            continue
        crop = inp0[rows.min():rows.max()+1, cols.min():cols.max()+1]
        if grids_match(crop, out0):
            ok = True
            for ex in train[1:]:
                i = np.array(ex['input'])
                o = np.array(ex['output'])
                r, c = np.where(i == color)
                if len(r) == 0 or not grids_match(i[r.min():r.max()+1, c.min():c.max()+1], o):
                    ok = False
                    break
            if ok:
                t = np.array(task['test'][0]['input'])
                r, c = np.where(t == color)
                if len(r) > 0:
                    return t[r.min():r.max()+1, c.min():c.max()+1].tolist(), f'crop_color_{color}'
    
    # Border removal
    for b in [1, 2]:
        ok = True
        for ex in train:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            ih, iw = inp.shape
            if ih - 2*b <= 0 or iw - 2*b <= 0:
                ok = False
                break
            if not grids_match(inp[b:ih-b, b:iw-b], out):
                ok = False
                break
        if ok:
            t = np.array(task['test'][0]['input'])
            return t[b:t.shape[0]-b, b:t.shape[1]-b].tolist(), f'remove_border_{b}'
    
    return None


# ═══════════════════════════════════════════════
# Symmetry completion
# ═══════════════════════════════════════════════

def solve_symmetry(task):
    train = task['train']
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    bg = find_bg(np.array(train[0]['input']))
    
    syms = [
        ('hmirror', lambda r, c, h, w: (r, w-1-c)),
        ('vmirror', lambda r, c, h, w: (h-1-r, c)),
        ('dmirror', lambda r, c, h, w: (h-1-r, w-1-c)),
    ]
    
    for sname, sfn in syms:
        ok = True
        for ex in train:
            inp, out = np.array(ex['input']), np.array(ex['output'])
            h, w = inp.shape
            pred = inp.copy()
            for r in range(h):
                for c in range(w):
                    if pred[r, c] == bg:
                        mr, mc = sfn(r, c, h, w)
                        if 0 <= mr < h and 0 <= mc < w and pred[mr, mc] != bg:
                            pred[r, c] = pred[mr, mc]
            if not np.array_equal(pred, out):
                ok = False
                break
        
        if ok:
            t = np.array(task['test'][0]['input'])
            h, w = t.shape
            pred = t.copy()
            for r in range(h):
                for c in range(w):
                    if pred[r, c] == bg:
                        mr, mc = sfn(r, c, h, w)
                        if 0 <= mr < h and 0 <= mc < w and pred[mr, mc] != bg:
                            pred[r, c] = pred[mr, mc]
            return pred.tolist(), f'sym:{sname}'
    
    return None


# ═══════════════════════════════════════════════
# Boolean grid operations (split and combine)
# ═══════════════════════════════════════════════

def solve_boolean(task):
    train = task['train']
    inp0, out0 = np.array(train[0]['input']), np.array(train[0]['output'])
    ih, iw = inp0.shape
    oh, ow = out0.shape
    bg = find_bg(inp0)
    
    splits = []
    if ih == 2 * oh and iw == ow:
        splits.append(('htop_hbot', lambda g: (g[:g.shape[0]//2], g[g.shape[0]//2:])))
    if iw == 2 * ow and ih == oh:
        splits.append(('vleft_vright', lambda g: (g[:, :g.shape[1]//2], g[:, g.shape[1]//2:])))
    if ih == oh and iw == ow:
        # Maybe contains separator
        for r in range(1, ih):
            if len(set(inp0[r].flatten().tolist())) == 1:
                sep_val = int(inp0[r, 0])
                top = inp0[:r]
                bot = inp0[r+1:]
                if top.shape == out0.shape and bot.shape == out0.shape:
                    splits.append((f'sep_h{r}', lambda g, r=r: (g[:r], g[r+1:])))
        for c in range(1, iw):
            if len(set(inp0[:, c].flatten().tolist())) == 1:
                left = inp0[:, :c]
                right = inp0[:, c+1:]
                if left.shape == out0.shape and right.shape == out0.shape:
                    splits.append((f'sep_v{c}', lambda g, c=c: (g[:, :c], g[:, c+1:])))
    
    ops = [
        ('or', lambda a, b, bg: np.where(a != bg, a, b)),
        ('and', lambda a, b, bg: np.where((a != bg) & (b != bg), a, bg)),
        ('xor', lambda a, b, bg: np.where((a != bg) ^ (b != bg), np.where(a != bg, a, b), bg)),
        ('a_minus_b', lambda a, b, bg: np.where((a != bg) & (b == bg), a, bg)),
        ('b_minus_a', lambda a, b, bg: np.where((b != bg) & (a == bg), b, bg)),
        ('b_over_a', lambda a, b, bg: np.where(b != bg, b, a)),
    ]
    
    for sname, sfn in splits:
        for oname, ofn in ops:
            ok = True
            for ex in train:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                bgi = find_bg(inp)
                try:
                    a, b = sfn(inp)
                    pred = ofn(a, b, bgi)
                    if not grids_match(pred, out):
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            
            if ok:
                t = np.array(task['test'][0]['input'])
                bgt = find_bg(t)
                a, b = sfn(t)
                return ofn(a, b, bgt).tolist(), f'bool:{sname}:{oname}'
    
    return None


# ═══════════════════════════════════════════════
# Row/col aggregation
# ═══════════════════════════════════════════════

def solve_rowcol(task):
    train = task['train']
    out0 = np.array(train[0]['output'])
    inp0 = np.array(train[0]['input'])
    
    # Output is 1 row
    if out0.shape[0] == 1 and out0.shape[1] == inp0.shape[1]:
        bg = find_bg(inp0)
        
        def row_or(inp):
            h, w = inp.shape
            bgi = find_bg(inp)
            result = np.full((1, w), bgi, dtype=int)
            for c in range(w):
                col = inp[:, c]
                nb = col[col != bgi]
                if len(nb) > 0:
                    result[0, c] = Counter(nb.tolist()).most_common(1)[0][0]
            return result
        
        ok = True
        for ex in train:
            if not grids_match(row_or(np.array(ex['input'])), np.array(ex['output'])):
                ok = False
                break
        if ok:
            return row_or(np.array(task['test'][0]['input'])).tolist(), 'row_or'
    
    # Output is 1 col
    if out0.shape[1] == 1 and out0.shape[0] == inp0.shape[0]:
        def col_or(inp):
            h, w = inp.shape
            bgi = find_bg(inp)
            result = np.full((h, 1), bgi, dtype=int)
            for r in range(h):
                row = inp[r, :]
                nb = row[row != bgi]
                if len(nb) > 0:
                    result[r, 0] = Counter(nb.tolist()).most_common(1)[0][0]
            return result
        
        ok = True
        for ex in train:
            if not grids_match(col_or(np.array(ex['input'])), np.array(ex['output'])):
                ok = False
                break
        if ok:
            return col_or(np.array(task['test'][0]['input'])).tolist(), 'col_or'
    
    # Row/col sort
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    sorts = [
        ('sort_rows_sum', lambda inp: inp[np.argsort([np.sum(inp[r]) for r in range(inp.shape[0])])]),
        ('sort_rows_sum_desc', lambda inp: inp[np.argsort([-np.sum(inp[r]) for r in range(inp.shape[0])])]),
        ('sort_rows_nz', lambda inp: inp[np.argsort([np.count_nonzero(inp[r]) for r in range(inp.shape[0])])]),
        ('reverse_rows', lambda inp: inp[::-1]),
    ]
    
    for sname, sfn in sorts:
        ok = True
        for ex in train:
            inp, out = np.array(ex['input']), np.array(ex['output'])
            try:
                if not grids_match(sfn(inp), out):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            return sfn(np.array(task['test'][0]['input'])).tolist(), sname
    
    return None


# ═══════════════════════════════════════════════
# Flood-fill propagation
# ═══════════════════════════════════════════════

def solve_flood(task):
    train = task['train']
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    bg = find_bg(np.array(train[0]['input']))
    
    for n_thresh in [1, 2, 3]:
        def propagate(grid, thresh, max_iter=100):
            h, w = grid.shape
            bgi = find_bg(grid)
            cur = grid.copy()
            for _ in range(max_iter):
                new = cur.copy()
                changed = False
                for r in range(h):
                    for c in range(w):
                        if cur[r, c] != bgi:
                            continue
                        n4 = []
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < h and 0 <= nc < w and cur[nr, nc] != bgi:
                                n4.append(int(cur[nr, nc]))
                        if len(n4) >= thresh:
                            new[r, c] = Counter(n4).most_common(1)[0][0]
                            changed = True
                cur = new
                if not changed:
                    break
            return cur
        
        ok = True
        for ex in train:
            pred = propagate(np.array(ex['input']), n_thresh)
            if not np.array_equal(pred, np.array(ex['output'])):
                ok = False
                break
        if ok:
            return propagate(np.array(task['test'][0]['input']), n_thresh).tolist(), f'flood_{n_thresh}'
    
    return None


# ═══════════════════════════════════════════════
# Counting output
# ═══════════════════════════════════════════════

def solve_minority_removal(task):
    """Remove minority-colored cells."""
    train = task['train']
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    bg = find_bg(np.array(train[0]['input']))
    
    # Strategy: remove isolated cells (no same-color 4-neighbor)
    def remove_isolated(grid):
        h, w = grid.shape
        bgi = find_bg(grid)
        result = grid.copy()
        for r in range(h):
            for c in range(w):
                if grid[r, c] == bgi:
                    continue
                val = int(grid[r, c])
                has_same = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == val:
                        has_same = True
                        break
                if not has_same:
                    result[r, c] = bgi
        return result
    
    ok = True
    for ex in train:
        if not np.array_equal(remove_isolated(np.array(ex['input'])), np.array(ex['output'])):
            ok = False
            break
    if ok:
        return remove_isolated(np.array(task['test'][0]['input'])).tolist(), 'remove_isolated'
    
    # Strategy: remove the least frequent non-bg color
    def remove_minority(grid):
        bgi = find_bg(grid)
        colors = Counter(grid[grid != bgi].flatten().tolist())
        if not colors:
            return grid.copy()
        minority = colors.most_common()[-1][0]
        result = grid.copy()
        result[grid == minority] = bgi
        return result
    
    ok = True
    for ex in train:
        if not np.array_equal(remove_minority(np.array(ex['input'])), np.array(ex['output'])):
            ok = False
            break
    if ok:
        return remove_minority(np.array(task['test'][0]['input'])).tolist(), 'remove_minority'
    
    # Strategy: remove specific color (learned from examples)
    inp0, out0 = np.array(train[0]['input']), np.array(train[0]['output'])
    diff = inp0 != out0
    if diff.any():
        removed_colors = set(inp0[diff].flatten().tolist()) - set(out0[diff].flatten().tolist()) - {bg}
        for rc in removed_colors:
            def remove_color(grid, c=rc):
                result = grid.copy()
                result[grid == c] = find_bg(grid)
                return result
            
            ok = True
            for ex in train:
                if not np.array_equal(remove_color(np.array(ex['input'])), np.array(ex['output'])):
                    ok = False
                    break
            if ok:
                return remove_color(np.array(task['test'][0]['input'])).tolist(), f'remove_color_{rc}'
    
    return None


def solve_fill_enclosed(task):
    """Fill enclosed bg regions."""
    train = task['train']
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    bg = find_bg(np.array(train[0]['input']))
    
    def fill_enclosed(grid, fill_mode='border'):
        h, w = grid.shape
        bgi = find_bg(grid)
        result = grid.copy()
        
        # Find outside bg using flood fill from edges
        outside = np.zeros((h, w), dtype=bool)
        queue = []
        for r in range(h):
            for c in [0, w-1]:
                if grid[r, c] == bgi and not outside[r, c]:
                    outside[r, c] = True
                    queue.append((r, c))
        for c in range(w):
            for r in [0, h-1]:
                if grid[r, c] == bgi and not outside[r, c]:
                    outside[r, c] = True
                    queue.append((r, c))
        
        while queue:
            r, c = queue.pop()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and not outside[nr, nc] and grid[nr, nc] == bgi:
                    outside[nr, nc] = True
                    queue.append((nr, nc))
        
        # Fill interior bg cells
        interior = (grid == bgi) & ~outside
        if not interior.any():
            return result
        
        if fill_mode == 'border':
            # Find regions and fill with border color
            labeled, n = ndimage.label(interior)
            for i in range(1, n+1):
                region = labeled == i
                rows, cols = np.where(region)
                border_colors = []
                for r, c in zip(rows, cols):
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != bgi:
                            border_colors.append(int(grid[nr, nc]))
                if border_colors:
                    fill_c = Counter(border_colors).most_common(1)[0][0]
                    result[region] = fill_c
        
        return result
    
    ok = True
    for ex in train:
        if not np.array_equal(fill_enclosed(np.array(ex['input'])), np.array(ex['output'])):
            ok = False
            break
    if ok:
        return fill_enclosed(np.array(task['test'][0]['input'])).tolist(), 'fill_enclosed'
    
    # Try: fill all interior with a single learned color
    inp0, out0 = np.array(train[0]['input']), np.array(train[0]['output'])
    diff = inp0 != out0
    if diff.any():
        fill_colors = set(out0[diff].flatten().tolist())
        if len(fill_colors) == 1:
            fill_c = list(fill_colors)[0]
            
            def fill_with_color(grid, c=fill_c):
                h, w = grid.shape
                bgi = find_bg(grid)
                result = grid.copy()
                outside = np.zeros((h, w), dtype=bool)
                queue = []
                for r in range(h):
                    for ci in [0, w-1]:
                        if grid[r, ci] == bgi and not outside[r, ci]:
                            outside[r, ci] = True
                            queue.append((r, ci))
                for ci in range(w):
                    for r in [0, h-1]:
                        if grid[r, ci] == bgi and not outside[r, ci]:
                            outside[r, ci] = True
                            queue.append((r, ci))
                while queue:
                    r, ci = queue.pop()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and not outside[nr, nc] and grid[nr, nc] == bgi:
                            outside[nr, nc] = True
                            queue.append((nr, nc))
                interior = (grid == bgi) & ~outside
                result[interior] = c
                return result
            
            ok = True
            for ex in train:
                if not np.array_equal(fill_with_color(np.array(ex['input'])), np.array(ex['output'])):
                    ok = False
                    break
            if ok:
                return fill_with_color(np.array(task['test'][0]['input'])).tolist(), f'fill_enclosed_{fill_c}'
    
    return None


def solve_per_object_transform(task):
    """Apply a per-object transform: recolor based on object property."""
    train = task['train']
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    bg = find_bg(np.array(train[0]['input']))
    
    # Check: does each object get recolored based on its size?
    rules_data = []
    for ex in train:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        objs = []
        labeled, n = ndimage.label(inp != bg)
        for i in range(1, n+1):
            mask = labeled == i
            size = int(mask.sum())
            old_color = int(Counter(inp[mask].flatten().tolist()).most_common(1)[0][0])
            # What color in output at same positions?
            out_vals = out[mask]
            new_color = int(Counter(out_vals.flatten().tolist()).most_common(1)[0][0])
            objs.append({'size': size, 'old_color': old_color, 'new_color': new_color, 'mask': mask})
        rules_data.append(objs)
    
    # Try: size → new_color mapping
    size_map = {}
    ok = True
    for objs in rules_data:
        for o in objs:
            k = o['size']
            v = o['new_color']
            if k in size_map and size_map[k] != v:
                ok = False
                break
            size_map[k] = v
        if not ok:
            break
    
    if ok and size_map:
        test_inp = np.array(task['test'][0]['input'])
        result = test_inp.copy()
        labeled, n = ndimage.label(test_inp != bg)
        for i in range(1, n+1):
            mask = labeled == i
            size = int(mask.sum())
            if size in size_map:
                result[mask] = size_map[size]
            # else keep original
        
        gt = task['test'][0].get('output')
        if gt and grids_match(result, gt):
            return result.tolist(), 'obj_size_recolor'
    
    return None


def solve_subgrid_extract(task):
    """Output is a subgrid of the input, extracted based on some rule."""
    train = task['train']
    
    # All outputs must be same size
    out_shapes = [np.array(ex['output']).shape for ex in train]
    if len(set(out_shapes)) != 1:
        return None
    oh, ow = out_shapes[0]
    
    # Find extraction positions for each training example
    positions = []
    for ex in train:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        ih, iw = inp.shape
        if oh > ih or ow > iw:
            return None
        
        found = None
        for r in range(ih - oh + 1):
            for c in range(iw - ow + 1):
                if np.array_equal(inp[r:r+oh, c:c+ow], out):
                    found = (r, c)
                    break
            if found:
                break
        
        if found is None:
            return None
        positions.append(found)
    
    # Rule 1: fixed position
    if len(set(positions)) == 1:
        r0, c0 = positions[0]
        test_inp = np.array(task['test'][0]['input'])
        if r0 + oh <= test_inp.shape[0] and c0 + ow <= test_inp.shape[1]:
            return test_inp[r0:r0+oh, c0:c0+ow].tolist(), f'extract_fixed_{r0}_{c0}'
    
    # Rule 2: extract around unique/marker color
    for mc in range(10):
        ok = True
        for i, ex in enumerate(train):
            inp = np.array(ex['input'])
            r0, c0 = positions[i]
            # Does this marker color exist in the output region?
            region = inp[r0:r0+oh, c0:c0+ow]
            if mc not in region:
                ok = False
                break
            # Does it exist ONLY in the output region?
            mask = np.ones_like(inp, dtype=bool)
            mask[r0:r0+oh, c0:c0+ow] = False
            if mc in inp[mask]:
                ok = False
                break
        
        if ok:
            # Apply: find the unique color in test input
            test_inp = np.array(task['test'][0]['input'])
            mc_positions = list(zip(*np.where(test_inp == mc)))
            if mc_positions:
                # Extract region containing this marker
                for mr, mcc in mc_positions:
                    for r0 in range(max(0, mr-oh+1), min(mr+1, test_inp.shape[0]-oh+1)):
                        for c0 in range(max(0, mcc-ow+1), min(mcc+1, test_inp.shape[1]-ow+1)):
                            crop = test_inp[r0:r0+oh, c0:c0+ow]
                            if mc in crop:
                                return crop.tolist(), f'extract_marker_{mc}'
    
    # Rule 3: extract around object with specific property
    bg = find_bg(np.array(train[0]['input']))
    
    # Check: output region contains the most/least/unique objects
    prop_extractors = [
        ('most_unique_colors', lambda region, bg: len(set(region.flatten().tolist()) - {bg})),
        ('most_nonbg', lambda region, bg: np.count_nonzero(region != bg)),
        ('least_nonbg', lambda region, bg: -np.count_nonzero(region != bg)),
    ]
    
    for pname, pfn in prop_extractors:
        ok = True
        for i, ex in enumerate(train):
            inp = np.array(ex['input'])
            r0, c0 = positions[i]
            ih, iw = inp.shape
            
            target_val = pfn(inp[r0:r0+oh, c0:c0+ow], bg)
            
            # Check all possible positions — is target the max?
            all_vals = []
            for r in range(ih - oh + 1):
                for c in range(iw - ow + 1):
                    all_vals.append((pfn(inp[r:r+oh, c:c+ow], bg), r, c))
            all_vals.sort(reverse=True)
            
            if all_vals[0][0] != target_val:
                ok = False
                break
        
        if ok:
            test_inp = np.array(task['test'][0]['input'])
            tih, tiw = test_inp.shape
            bgt = find_bg(test_inp)
            best = None
            best_val = float('-inf')
            for r in range(tih - oh + 1):
                for c in range(tiw - ow + 1):
                    v = pfn(test_inp[r:r+oh, c:c+ow], bgt)
                    if v > best_val:
                        best_val = v
                        best = (r, c)
            if best:
                r0, c0 = best
                return test_inp[r0:r0+oh, c0:c0+ow].tolist(), f'extract_{pname}'
    
    return None


def solve_counting(task):
    train = task['train']
    out0 = np.array(train[0]['output'])
    
    if out0.size != 1:
        return None
    
    rules = [
        ('n_colors', lambda g: len(set(g.flatten().tolist()))),
        ('n_colors_nonbg', lambda g: len(set(g.flatten().tolist()) - {find_bg(g)})),
        ('n_objects', lambda g: ndimage.label(g != find_bg(g))[1]),
        ('max_val', lambda g: int(g.max())),
        ('n_nonzero', lambda g: int(np.count_nonzero(g))),
    ]
    
    for rname, rfn in rules:
        ok = True
        for ex in train:
            try:
                pred = rfn(np.array(ex['input']))
                gt = int(np.array(ex['output']).flatten()[0])
                if pred != gt:
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            return [[rfn(np.array(task['test'][0]['input']))]], rname
    
    return None


# ═══════════════════════════════════════════════
# MAIN SOLVE
# ═══════════════════════════════════════════════

def solve_all(task):
    strategies = [
        solve_nb_lut,      # Exact neighborhood match
        solve_abstract,    # Abstract per-cell features
        solve_colormap,    # Simple color remapping
        solve_geometric,   # Flips, rotations, transposes
        solve_scale,       # Scale up/down
        solve_tile,        # Tiling/repeat
        solve_crop,        # Crop/extract/border removal
        solve_partition,   # Grid partitioning
        solve_boolean,     # Boolean grid ops
        solve_symmetry,    # Symmetry completion
        solve_rowcol,      # Row/col operations
        solve_flood,       # Flood fill propagation
        solve_minority_removal,  # Remove isolated/minority cells
        solve_fill_enclosed,     # Fill enclosed regions
        solve_per_object_transform,  # Per-object recoloring
        solve_subgrid_extract,   # Extract subgrid from input
        solve_counting,    # Output = count/statistic
    ]
    
    for fn in strategies:
        try:
            if fn == solve_nb_lut:
                for r in [1, 2]:
                    result = fn(task, radius=r)
                    if result:
                        return result
            else:
                result = fn(task)
                if result:
                    return result
        except Exception:
            continue
    
    return None


def _worker(task_json, result_queue):
    task = json.loads(task_json)
    gt = task['test'][0].get('output')
    if gt is None:
        result_queue.put((False, 'no_gt'))
        return
    result = solve_all(task)
    if result:
        pred, method = result
        if grids_match(pred, gt):
            result_queue.put((True, method))
            return
    result_queue.put((False, 'none'))


def solve_w_timeout(task_json, timeout=45):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(task_json, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join(3)
        return False, 'timeout'
    if not q.empty():
        try:
            return q.get_nowait()
        except Exception:
            pass
    return False, 'error'


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    
    eval_dir = 'ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation'
    
    prev = set()
    for f in ['arc_agi1_eval_results.json', 'arc_emerged_eval_results.json']:
        if os.path.exists(f):
            d = json.load(open(f))
            prev |= set(d.get('solved_ids', []))
    
    log.info("=" * 60)
    log.info("⚡ MEGA ARC SOLVER — ALL STRATEGIES")
    log.info(f"Previously solved: {len(prev)}/400")
    log.info("=" * 60)
    
    solved = []
    methods = {}
    timeouts = 0
    total = 0
    t0 = time.time()
    
    for fn in sorted(os.listdir(eval_dir)):
        if not fn.endswith('.json'):
            continue
        tid = fn.replace('.json', '')
        total += 1
        
        task = json.load(open(os.path.join(eval_dir, fn)))
        ok, method = solve_w_timeout(json.dumps(task), timeout=45)
        
        if ok:
            solved.append(tid)
            methods[method] = methods.get(method, 0) + 1
            tag = "🆕" if tid not in prev else ""
            log.info(f"  ✅ {tid} [{method}] {tag}")
        elif method == 'timeout':
            timeouts += 1
        
        if total % 100 == 0:
            wall = time.time() - t0
            new = len([t for t in solved if t not in prev])
            log.info(f"  [{total}/400] {len(solved)} solved (+{new} new), {timeouts} timeouts | {wall:.0f}s")
    
    wall = time.time() - t0
    new_ids = sorted([t for t in solved if t not in prev])
    combined = prev | set(solved)
    
    log.info(f"\n{'='*60}")
    log.info(f"📊 MEGA SOLVER RESULTS")
    log.info(f"{'='*60}")
    log.info(f"  This solver:    {len(solved)}/{total}")
    log.info(f"  Timeouts:       {timeouts}")
    log.info(f"  🆕 New:         {len(new_ids)}")
    log.info(f"  ★ COMBINED:     {len(combined)}/400 ({len(combined)/4:.1f}%)")
    log.info(f"  Time:           {wall:.0f}s")
    log.info(f"  Methods:        {methods}")
    if new_ids:
        log.info(f"  🆕 New IDs:     {new_ids}")
    
    out = {
        'solver_solved': len(solved),
        'new': len(new_ids),
        'combined': len(combined),
        'pct': round(len(combined)/4, 1),
        'solved_ids': sorted(solved),
        'new_ids': new_ids,
        'combined_ids': sorted(combined),
        'methods': methods,
        'timeouts': timeouts,
        'time_s': round(wall, 1),
    }
    with open('arc_mega_eval_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    log.info(f"  → Saved arc_mega_eval_results.json")


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    main()
