"""
ARC-AGI Task 40494e2f Solver

Pattern:
1. Input contains rectangles (large filled regions) and a stamp pattern (small cross/plus)
2. The stamp pattern is placed inside each rectangle (possibly multiple times)
3. The original stamp is erased from the input
4. Colors are detected by frequency (not hardcoded)

This is the sophisticated version from arc_40494e2f_solver.py
"""

import json
import math
from collections import Counter, deque


def transform(grid):
    """Transform grid by stamping pattern into rectangles."""
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # 1. Find colors: bg (most common), rect_color (2nd), marker_color (3rd)
    flat = [grid[r][c] for r in range(H) for c in range(W)]
    by_freq = Counter(flat).most_common()
    bg = by_freq[0][0]
    rect_color = by_freq[1][0]
    marker_color = by_freq[2][0] if len(by_freq) > 2 else None
    if marker_color is None:
        return out

    # 2. Find rectangles (connected components of rect_color, bbox area >= 15)
    visited = [[False]*W for _ in range(H)]
    rectangles = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == rect_color and not visited[r][c]:
                q = deque([(r, c)]); visited[r][c] = True; cells = []
                while q:
                    cr, cc = q.popleft(); cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<H and 0<=nc<W and not visited[nr][nc] and grid[nr][nc]==rect_color:
                            visited[nr][nc] = True; q.append((nr, nc))
                rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
                t, b, l, ri = min(rs), max(rs), min(cs), max(cs)
                bbox_area = (b-t+1)*(ri-l+1)
                # Must be a filled rectangle (all cells filled) AND large enough
                if bbox_area == len(cells) and bbox_area >= 15:
                    rectangles.append((t, b, l, ri))

    # Build set of rect bbox cells
    rect_interior = set()
    for t, b, l, ri in rectangles:
        for r in range(t, b+1):
            for c in range(l, ri+1):
                rect_interior.add((r, c))

    # 3. Stamp cells (non-bg outside all rect bboxes)
    stamp_cells = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and (r, c) not in rect_interior:
                stamp_cells[(r, c)] = grid[r][c]
    if not stamp_cells:
        return out

    # 4. Find stamp center using bbox center approach
    spos = list(stamp_cells.keys())
    s_rmin, s_rmax = min(p[0] for p in spos), max(p[0] for p in spos)
    s_cmin, s_cmax = min(p[1] for p in spos), max(p[1] for p in spos)
    r_mid, c_mid = (s_rmin + s_rmax) / 2.0, (s_cmin + s_cmax) / 2.0

    candidates = set()
    for rf in [math.floor(r_mid), math.ceil(r_mid)]:
        for cf in [math.floor(c_mid), math.ceil(c_mid)]:
            if 0 <= rf < H and 0 <= cf < W:
                candidates.add((rf, cf))

    best_center, best_score = None, (-1, -1, -1, -1, -1)
    for r, c in sorted(candidates):
        nb = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)] if (r+dr, c+dc) in stamp_cells)
        is_nm = 1 if grid[r][c] != marker_color else 0
        dist = abs(r - r_mid) + abs(c - c_mid)
        score = (is_nm, nb, -dist, -r, -c)
        if score > best_score:
            best_score = score; best_center = (r, c)

    center_r, center_c = best_center

    # 5. Build stamp pattern relative to center
    stamp_pattern = {}
    for (r, c), color in stamp_cells.items():
        stamp_pattern[(r - center_r, c - center_c)] = color
    stamp_pattern[(0, 0)] = grid[center_r][center_c]

    # 6. Detect extending arms and lengths
    def extends(dr, dc):
        return stamp_pattern.get((dr, dc)) == marker_color and stamp_pattern.get((2*dr, 2*dc)) == marker_color

    ext_up, ext_down = extends(-1, 0), extends(1, 0)
    ext_left, ext_right = extends(0, -1), extends(0, 1)

    def arm_len_fn(dr, dc):
        n = 0
        for k in range(1, 50):
            if stamp_pattern.get((k*dr, k*dc)) == marker_color: n = k
            else: break
        return n

    up_len = arm_len_fn(-1, 0) if ext_up else 0
    down_len = arm_len_fn(1, 0) if ext_down else 0
    left_len = arm_len_fn(0, -1) if ext_left else 0
    right_len = arm_len_fn(0, 1) if ext_right else 0
    arm_len = max(up_len, down_len, left_len, right_len, 1)

    marker_offsets = [(dr, dc) for (dr, dc), v in stamp_pattern.items() if v == marker_color]

    # Body extents (all stamp cells, not just markers)
    all_drs = [dr for dr, dc in stamp_pattern]
    all_dcs = [dc for dr, dc in stamp_pattern]
    body_up = abs(min(all_drs))    # distance above center
    body_down = max(all_drs)       # distance below center
    body_left = abs(min(all_dcs))  # distance left of center
    body_right = max(all_dcs)      # distance right of center
    body_height = body_up + body_down + 1
    body_width = body_left + body_right + 1

    # d1 marker checks (for non-extending semi-arms)
    up_d1_marker = stamp_pattern.get((-1, 0)) == marker_color
    down_d1_marker = stamp_pattern.get((1, 0)) == marker_color
    left_d1_marker = stamp_pattern.get((0, -1)) == marker_color
    right_d1_marker = stamp_pattern.get((0, 1)) == marker_color

    # 7-8. Process each rectangle
    for rt, rb, rl, rr in rectangles:
        rH, rW = rb - rt + 1, rr - rl + 1

        holes = [(r, c) for r in range(rt, rb+1) for c in range(rl, rr+1) if grid[r][c] != rect_color]

        if holes:
            centers = holes
        else:
            centers = _no_hole_centers(
                rt, rb, rl, rr, rH, rW,
                ext_up, ext_down, ext_left, ext_right,
                up_len, down_len, left_len, right_len, arm_len,
                up_d1_marker, down_d1_marker, left_d1_marker, right_d1_marker,
                body_up, body_down, body_left, body_right, body_height, body_width,
                stamp_pattern, marker_color, center_r, center_c)

        for cr_pos, cc_pos in centers:
            _place(out, cr_pos, cc_pos, marker_offsets, marker_color,
                   ext_up, ext_down, ext_left, ext_right, rt, rb, rl, rr)

    # 9. Erase stamp
    for (r, c) in stamp_cells:
        out[r][c] = bg
    return out


def _place(out, cr, cc, marker_offsets, marker_color,
           ext_up, ext_down, ext_left, ext_right, rt, rb, rl, rr):
    for dr, dc in marker_offsets:
        nr, nc = cr+dr, cc+dc
        if rt <= nr <= rb and rl <= nc <= rr:
            out[nr][nc] = marker_color
    if ext_up:
        for r in range(rt, cr): out[r][cc] = marker_color
    if ext_down:
        for r in range(cr+1, rb+1): out[r][cc] = marker_color
    if ext_left:
        for c in range(rl, cc): out[cr][c] = marker_color
    if ext_right:
        for c in range(cc+1, rr+1): out[cr][c] = marker_color


def _no_hole_centers(rt, rb, rl, rr, rH, rW,
                     ext_up, ext_down, ext_left, ext_right,
                     up_len, down_len, left_len, right_len, arm_len,
                     up_d1_m, down_d1_m, left_d1_m, right_d1_m,
                     body_up, body_down, body_left, body_right, body_h, body_w,
                     stamp_pattern, marker_color, stamp_cr, stamp_cc):
    # === VERTICAL POSITIONS ===
    if ext_down and not ext_up:
        v_start = arm_len; v_period = arm_len + 1
        row_pos = _gen_positions(v_start, v_period, rH)
        max_rows = max(1, rH // (2 * v_period))
        row_pos = row_pos[:max_rows]
    elif ext_up and not ext_down:
        v_start = rH - 1 - arm_len; v_period = arm_len + 1
        row_pos = list(reversed(_gen_positions_rev(v_start, v_period)))
        max_rows = max(1, rH // (2 * v_period))
        row_pos = row_pos[:max_rows]
    elif ext_up and ext_down:
        v_period = 2 * arm_len
        row_pos = _gen_positions(arm_len, v_period, rH)
        max_rows = max(1, rH // (2 * (arm_len + 1)))
        row_pos = row_pos[:max_rows]
    elif up_d1_m and not down_d1_m and not ext_up and not ext_down:
        # Asymmetric: UP has semi-arm, DOWN doesn't
        row_pos = [min(body_up + 2, rH - 1 - body_down)]
    elif down_d1_m and not up_d1_m and not ext_up and not ext_down:
        # Asymmetric: DOWN has semi-arm, UP doesn't
        row_pos = [max(rH - 1 - body_down - 2, body_up)]
    else:
        # Symmetric or no d1 markers: center
        row_pos = [math.ceil((rH - 1) / 2)]

    if not row_pos:
        row_pos = [math.ceil((rH - 1) / 2)]

    # === HORIZONTAL POSITIONS ===
    if ext_left and ext_right:
        h_period = max(left_len + right_len, 2)
        h_start = max(left_len, right_len)
        col_pos = _gen_positions(h_start, h_period, rW)
    elif ext_left and not ext_right:
        col_pos = [_bounce_and_flip(stamp_cc, rl, rr, rW, arm_len, 'left')]
    elif ext_right and not ext_left:
        col_pos = [_bounce_and_flip(stamp_cc, rl, rr, rW, arm_len, 'right')]
    elif not ext_left and not ext_right:
        # Check snug condition for horizontal shift
        if body_h >= rH - 1:
            rect_center_c = (rl + rr) / 2.0
            if stamp_cc > rr:  # stamp right of rect
                col_pos = [rW - 1 - body_right]
            elif stamp_cc < rl:  # stamp left of rect
                col_pos = [body_left]
            else:  # stamp inside rect column range
                col_pos = [math.ceil((rW - 1) / 2)]
        else:
            col_pos = [math.ceil((rW - 1) / 2)]
    else:
        col_pos = [math.ceil((rW - 1) / 2)]

    if not col_pos:
        col_pos = [math.ceil((rW - 1) / 2)]

    # Clamp positions
    row_pos = [max(0, min(rH-1, r)) for r in row_pos]
    col_pos = [max(0, min(rW-1, c)) for c in col_pos]

    # === COMBINE ===
    if ext_left and ext_right and len(row_pos) == 1 and len(col_pos) >= 2:
        # Single row: use midpoint of column positions
        mid_col = (col_pos[0] + col_pos[-1]) // 2
        return [(rt + row_pos[0], rl + mid_col)]
    elif ext_left and ext_right and len(row_pos) > 1 and len(col_pos) >= 2:
        # Multiple rows: checkerboard (i+j odd)
        centers = []
        for i, rp in enumerate(row_pos):
            for j, cp in enumerate(col_pos):
                if (i + j) % 2 == 1:
                    centers.append((rt + rp, rl + cp))
        if not centers:
            for i, rp in enumerate(row_pos):
                for j, cp in enumerate(col_pos):
                    if (i + j) % 2 == 0:
                        centers.append((rt + rp, rl + cp))
        return centers
    else:
        return [(rt + rp, rl + cp) for rp in row_pos for cp in col_pos]


def _gen_positions(start, period, limit):
    pos = []
    v = start
    while v < limit:
        pos.append(v); v += period
    return pos

def _gen_positions_rev(start, period):
    pos = []
    v = start
    while v >= 0:
        pos.append(v); v -= period
    return pos

def _bounce_and_flip(stamp_col, rect_left, rect_right, rW, arm_len, ext_dir):
    offset = stamp_col - rect_left
    period = 2 * (rW - 1) if rW > 1 else 1
    offset = offset % period
    if offset < 0: offset += period
    if offset > rW - 1:
        offset = period - offset
    offset = max(1, min(rW - 2, offset))
    # Flip if stamp is on the same side as the extending direction
    rect_center_c = (rect_left + rect_right) / 2.0
    if ext_dir == 'left' and stamp_col < rect_center_c:
        offset = rW - offset
    elif ext_dir == 'right' and stamp_col > rect_center_c:
        offset = rW - offset
    offset = max(1, min(rW - 2, offset))
    return offset


def test_solver():
    """Test the solver on training examples."""
    with open('40494e2f.json') as f:
        data = json.load(f)
    
    all_pass = True
    
    for i, pair in enumerate(data['train']):
        inp = [row[:] for row in pair['input']]
        expected = pair['output']
        result = transform(inp)
        
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            all_pass = False
            # Show first few differences
            diffs = []
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if r < len(result) and c < len(result[0]):
                        if result[r][c] != expected[r][c]:
                            diffs.append((r, c, result[r][c], expected[r][c]))
            
            print(f"  {len(diffs)} differences found")
            for r, c, got, exp in diffs[:10]:
                print(f"    ({r},{c}): got {got}, expected {exp}")
            if len(diffs) > 10:
                print(f"    ... and {len(diffs) - 10} more")
    
    return all_pass


if __name__ == '__main__':
    if test_solver():
        print("\n✓ All training examples pass!")
    else:
        print("\n✗ Some tests failed")
