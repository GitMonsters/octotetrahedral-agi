import json
import numpy as np
from collections import Counter

def transform(input_grid):
    grid = np.array(input_grid)
    H, W = grid.shape
    bg = Counter(grid.flatten()).most_common(1)[0][0]
    
    # Find non-bg pixels (at corners)
    pixels = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] != bg:
                pixels.append((r, c, int(grid[r, c])))
    
    peak_dr = min((H - 1) // 2, (W - 1) // 2)
    has_diff_color = len(set(c for _, _, c in pixels)) > 1
    
    # Growth direction for each pixel
    def growth_dir(pr, pc):
        dr_s = 1 if pr <= (H-1)/2 else -1
        dc_s = 1 if pc <= (W-1)/2 else -1
        if pr == 0: dr_s = 1
        elif pr == H-1: dr_s = -1
        if pc == 0: dc_s = 1
        elif pc == W-1: dc_s = -1
        return dr_s, dc_s
    
    pinfo = []
    for pr, pc, col in pixels:
        dr_s, dc_s = growth_dir(pr, pc)
        pinfo.append({'r': pr, 'c': pc, 'color': col, 'dr': dr_s, 'dc': dc_s})
    
    n = len(pixels)
    
    # Step 1: Compute pw for each pixel from each axis
    pw_from_col = [peak_dr + 1] * n  # from same-column competition
    pw_from_row = [peak_dr + 1] * n  # from same-row competition
    
    col_partner = [None] * n  # index of same-col partner
    row_partner = [None] * n
    col_D = [0] * n
    row_D = [0] * n
    col_is_first = [False] * n
    row_is_first = [False] * n
    col_same_color = [False] * n
    row_same_color = [False] * n
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            pi, pj = pinfo[i], pinfo[j]
            # Same column
            if pi['c'] == pj['c']:
                D = abs(pi['r'] - pj['r'])
                col_partner[i] = j
                col_D[i] = D
                col_is_first[i] = (pi['r'] < pj['r'])
                col_same_color[i] = (pi['color'] == pj['color'])
                if pi['color'] == pj['color']:
                    pw_from_col[i] = min(pw_from_col[i], D // 2)
                else:
                    if pi['r'] < pj['r']:
                        pw_from_col[i] = min(pw_from_col[i], (D + 1) // 2)
                    else:
                        pw_from_col[i] = min(pw_from_col[i], D // 2)
            # Same row
            if pi['r'] == pj['r']:
                D = abs(pi['c'] - pj['c'])
                row_partner[i] = j
                row_D[i] = D
                row_is_first[i] = (pi['c'] < pj['c'])
                row_same_color[i] = (pi['color'] == pj['color'])
                if pi['color'] == pj['color']:
                    pw_from_row[i] = min(pw_from_row[i], D // 2)
                else:
                    if pi['c'] < pj['c']:
                        pw_from_row[i] = min(pw_from_row[i], (D + 1) // 2)
                    else:
                        pw_from_row[i] = min(pw_from_row[i], D // 2)
    
    # Cap at peak_dr + 1
    for i in range(n):
        pw_from_col[i] = min(pw_from_col[i], peak_dr + 1)
        pw_from_row[i] = min(pw_from_row[i], peak_dr + 1)
    
    # Effective pw = min of all axis pws
    pw = [min(pw_from_col[i], pw_from_row[i]) for i in range(n)]
    
    # Redistribute: if a pixel's effective pw < its axis pw, partner gets surplus
    for iteration in range(5):
        changed = False
        for i in range(n):
            # Check column axis
            j = col_partner[i]
            if j is not None and not col_same_color[i]:
                D = col_D[i]
                # Partner j's pw on this axis should be D - pw[i]
                new_pw_col_j = min(D - pw[i], peak_dr + 1)
                if new_pw_col_j > pw_from_col[j]:
                    pw_from_col[j] = new_pw_col_j
                    new_pw_j = min(pw_from_col[j], pw_from_row[j])
                    if new_pw_j != pw[j]:
                        pw[j] = new_pw_j
                        changed = True
            # Check row axis
            j = row_partner[i]
            if j is not None and not row_same_color[i]:
                D = row_D[i]
                new_pw_row_j = min(D - pw[i], peak_dr + 1)
                if new_pw_row_j > pw_from_row[j]:
                    pw_from_row[j] = new_pw_row_j
                    new_pw_j = min(pw_from_col[j], pw_from_row[j])
                    if new_pw_j != pw[j]:
                        pw[j] = new_pw_j
                        changed = True
        if not changed:
            break
    
    # Step 2: Compute M (max dr), ALT_MAX, alt shape for each pixel
    M = [0] * n
    ALT_MAX = [0] * n
    alt_shape = ['flat'] * n  # 'flat', 'ceil', 'floor'
    
    for i in range(n):
        pi = pinfo[i]
        pwi = pw[i]
        
        # M from same-col competition
        if col_partner[i] is not None:
            M_col = (col_D[i] - 1) // 2
        else:
            M_col = 999
        
        # M from diamond max
        M_diamond = peak_dr + 2 * (pwi // 2)
        
        # M from grid bounds
        if pi['dr'] == 1:
            M_grid = H - 1 - pi['r']
        else:
            M_grid = pi['r']
        
        M[i] = min(M_col, M_diamond, M_grid)
        
        # ALT_MAX
        if pi['dc'] == 1:
            dc_extent = W - 1 - pi['c']
        else:
            dc_extent = pi['c']
        full_dc = 2 * (dc_extent // 2)
        
        if pwi >= peak_dr + 1:
            ALT_MAX[i] = full_dc
        elif has_diff_color:
            ALT_MAX[i] = 2 * ((pwi - 1) // 2)
        else:
            ALT_MAX[i] = full_dc
        
        # Alt shape
        has_same_axis_diff = False
        if col_partner[i] is not None and not col_same_color[i]:
            has_same_axis_diff = True
        if row_partner[i] is not None and not row_same_color[i]:
            has_same_axis_diff = True
        
        if ALT_MAX[i] < full_dc:
            alt_shape[i] = 'flat'
        elif has_same_axis_diff:
            alt_shape[i] = 'ceil'
        elif has_diff_color:
            alt_shape[i] = 'floor'
        else:
            alt_shape[i] = 'flat'
    
    # Step 3: Generate patterns
    # claims[r][c] = list of (color, 'solid'/'alt')
    solid_claims = {}
    alt_claims = {}
    
    for i in range(n):
        pi = pinfo[i]
        pwi = pw[i]
        mi = M[i]
        ami = ALT_MAX[i]
        
        for dr in range(mi + 1):
            r = pi['r'] + dr * pi['dr']
            if r < 0 or r >= H:
                continue
            
            # SOLID (even dr only)
            if dr % 2 == 0:
                growing = min(dr, pwi - 1)
                if dr > peak_dr:
                    k = (dr - peak_dr + 1) // 2
                    shrinking = pwi - 2 * k
                else:
                    shrinking = pwi
                solid_max = min(growing, shrinking)
                
                if solid_max >= 0:
                    for dc in range(solid_max + 1):
                        c = pi['c'] + dc * pi['dc']
                        if 0 <= c < W:
                            key = (r, c)
                            if key not in solid_claims:
                                solid_claims[key] = set()
                            solid_claims[key].add(pi['color'])
            
            # ALTERNATING
            alt_start = 2 * ((dr + 1) // 2)
            if alt_shape[i] == 'flat':
                alt_end = ami
            elif alt_shape[i] == 'ceil':
                alt_end = ami - 2 * ((dr + 1) // 2)
            elif alt_shape[i] == 'floor':
                alt_end = ami - 2 * (dr // 2)
            
            if alt_end >= alt_start and alt_end >= 0:
                for dc in range(alt_start, alt_end + 1, 2):
                    c = pi['c'] + dc * pi['dc']
                    if 0 <= c < W:
                        key = (r, c)
                        if key not in alt_claims:
                            alt_claims[key] = set()
                        alt_claims[key].add(pi['color'])
    
    # Step 4: Resolve conflicts
    output = np.full((H, W), bg, dtype=int)
    
    all_cells = set(solid_claims.keys()) | set(alt_claims.keys())
    for key in all_cells:
        r, c = key
        s_colors = solid_claims.get(key, set())
        a_colors = alt_claims.get(key, set())
        
        if s_colors:
            if len(s_colors) == 1:
                output[r, c] = list(s_colors)[0]
            # Multiple solid colors shouldn't happen; leave bg
        elif a_colors:
            if len(a_colors) == 1:
                output[r, c] = list(a_colors)[0]
            # Multiple alt colors → bg (conflict)
    
    return output.tolist()


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rearc_agent_solves/380c137c.json'
    with open(path) as f:
        data = json.load(f)
    
    for ti, ex in enumerate(data['train']):
        result = transform(ex['input'])
        expected = ex['output']
        match = result == expected
        if not match:
            diff = sum(1 for r in range(len(expected)) for c in range(len(expected[0]))
                      if result[r][c] != expected[r][c])
            print(f"Train {ti}: FAIL ({diff} diffs)")
            count = 0
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, exp {expected[r][c]}")
                        count += 1
                        if count >= 15: break
                if count >= 15: break
        else:
            print(f"Train {ti}: PASS")
