import json
from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    
    def find_stripes(axis):
        stripes = {}
        for i in range(H if axis == 'row' else W):
            line = [grid[i][j] if axis == 'row' else grid[j][i]
                    for j in range(W if axis == 'row' else H)]
            nonbg = [v for v in line if v != bg]
            if nonbg:
                mc = Counter(nonbg).most_common(1)[0]
                if mc[1] >= (W if axis == 'row' else H) * 0.6:
                    stripes[i] = mc[0]
        return stripes
    
    def fill_right(g, fc):
        result = [r[:] for r in g]
        for r in range(len(g)):
            for c in range(len(g[r])):
                if result[r][c] == fc:
                    for c2 in range(c, len(g[r])): result[r][c2] = fc
                    break
        return result
    
    def fill_left(g, fc):
        result = [r[:] for r in g]
        for r in range(len(g)):
            for c in range(len(g[r]) - 1, -1, -1):
                if result[r][c] == fc:
                    for c2 in range(c + 1): result[r][c2] = fc
                    break
        return result
    
    def fill_down(g, fc):
        result = [r[:] for r in g]
        for c in range(len(g[0])):
            for r in range(len(g)):
                if result[r][c] == fc:
                    for r2 in range(r, len(g)): result[r2][c] = fc
                    break
        return result
    
    def fill_up(g, fc):
        result = [r[:] for r in g]
        for c in range(len(g[0])):
            for r in range(len(g) - 1, -1, -1):
                if result[r][c] == fc:
                    for r2 in range(r + 1): result[r2][c] = fc
                    break
        return result
    
    row_stripes = find_stripes('row'); col_stripes = find_stripes('col')
    rs = sorted(row_stripes); cs = sorted(col_stripes)
    
    if len(rs) == 2 and len(cs) == 2:
        r1, r2 = rs; c1, c2 = cs
        extracted = [[grid[r1+i][c1+j] for j in range(c2-c1+1)] for i in range(r2-r1+1)]
        return fill_down(extracted, row_stripes[r2])
    
    elif len(rs) == 1 and len(cs) == 2:
        r1 = rs[0]; c1, c2 = cs
        w_out = c2 - c1 + 1
        rc = row_stripes[r1]; lc, rc2 = col_stripes[c1], col_stripes[c2]
        if rc == lc or rc == rc2:
            # Row stripe shares color with a col stripe → vertical fill (up), h_out = w_out-1
            h_out = w_out - 1
            extracted = [[grid[r1+i][c1+j] for j in range(w_out)] for i in range(h_out)]
            return fill_up(extracted, rc)
        else:
            # Row stripe unique → horizontal fill (right), h_out = w_out (square)
            h_out = w_out
            extracted = [[grid[r1+i][c1+j] for j in range(w_out)] for i in range(h_out)]
            return fill_right(extracted, rc2)
    
    elif len(rs) == 2 and len(cs) == 1:
        r1, r2 = rs; c0 = cs[0]
        h_out = r2 - r1 + 1
        fc = col_stripes[c0]
        rc_top, rc_bot = row_stripes[r1], row_stripes[r2]
        if fc == rc_top or fc == rc_bot:
            # Col stripe shares color with row stripe → horizontal fill
            w_out = h_out - 1
            if c0 + w_out < W:
                extracted = [[grid[r1+i][c0+j] for j in range(w_out+1)] for i in range(h_out)]
                return fill_right(extracted, fc)
            else:
                extracted = [[grid[r1+i][c0-w_out+j] for j in range(w_out+1)] for i in range(h_out)]
                return fill_left(extracted, fc)
        else:
            w_out = h_out
            if c0 + w_out <= W:
                extracted = [[grid[r1+i][c0+j] for j in range(w_out)] for i in range(h_out)]
                return fill_down(extracted, rc_bot)
            else:
                extracted = [[grid[r1+i][c0-w_out+1+j] for j in range(w_out)] for i in range(h_out)]
                return fill_up(extracted, rc_top)
    
    return grid

if __name__ == '__main__':
    with open('/Users/evanpieser/Desktop/ReArc45/re-arc_test_challenges-2026-04-05T23-26-25.json') as f:
        data = json.load(f)
    task = data['20f764f4']
    passed = 0
    for i, ex in enumerate(task['train']):
        pred = transform([row[:] for row in ex['input']])
        ok = pred == ex['output']
        print(f'Train {i}: {"PASS" if ok else "FAIL"}')
        if ok: passed += 1
    print(f'{passed}/{len(task["train"])} passing')
