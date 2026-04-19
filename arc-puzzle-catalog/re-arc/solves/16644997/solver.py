import json
from collections import Counter

def find_L_frame(K):
    Kh, Kw = len(K), len(K[0])
    frame_row = None
    frame_col = None
    for r_check, label in [(0,'top'),(Kh-1,'bottom')]:
        if len(set(K[r_check]))==1: frame_row=label; break
    for c_check, label in [(0,'left'),(Kw-1,'right')]:
        if len(set(K[r][c_check] for r in range(Kh)))==1: frame_col=label; break
    return frame_row, frame_col

def get_interior(K, frame_row, frame_col):
    Kh, Kw = len(K), len(K[0])
    r0 = 1 if frame_row=='top' else 0
    r1 = Kh-2 if frame_row=='bottom' else Kh-1
    c0 = 1 if frame_col=='left' else 0
    c1 = Kw-2 if frame_col=='right' else Kw-1
    return [K[r][c0:c1+1] for r in range(r0, r1+1)]

def is_multi_level(K):
    frame_row, frame_col = find_L_frame(K)
    if frame_row is None or frame_col is None: return False
    interior = get_interior(K, frame_row, frame_col)
    if not interior or not interior[0]: return False
    return len(set(v for row in interior for v in row)) >= 3

def transform(grid):
    R, C = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    
    rows_nb = [r for r in range(R) if any(grid[r][c]!=bg for c in range(C))]
    cols_nb = [c for c in range(C) if any(grid[r][c]!=bg for r in range(R))]
    r0, r1 = min(rows_nb), max(rows_nb)
    c0, c1 = min(cols_nb), max(cols_nb)
    K = [grid[r][c0:c1+1] for r in range(r0, r1+1)]
    Kh, Kw = len(K), len(K[0])
    Ro, Co = 2*R, 2*C

    if is_multi_level(K):
        offset = Ro % Kh
        def step(x): return x - (1 if x % Kh == offset else 0)
        seq = [K[Kh-1][(k + Kh - offset) % Kh] for k in range(Kh)]
        return [[seq[min(step(r), step(c)) % Kh] for c in range(Co)] for r in range(Ro)]
    else:
        N = Co // Kw
        at_top = (r0 == 0)
        at_left = (c0 == 0)
        use_anti_diag = (at_top and not at_left) or (not at_top and at_left)
        B_col = Kw-1 if use_anti_diag else 0
        out = []
        for r in range(Ro):
            row = []
            for c in range(Co):
                br, bc, rr, rc = r//Kh, c//Kw, r%Kh, c%Kw
                diag = br+bc if use_anti_diag else (br-bc)
                ref  = N-1  if use_anti_diag else 0
                if (use_anti_diag and diag==N-1) or (not use_anti_diag and br==bc):
                    row.append(K[rr][rc])
                elif (use_anti_diag and diag<N-1) or (not use_anti_diag and br<bc):
                    row.append(K[0][rc])
                else:
                    row.append(K[rr][B_col])
            out.append(row)
        return out

if __name__ == '__main__':
    with open('/Users/evanpieser/Desktop/ReArc45/re-arc_test_challenges-2026-04-05T23-26-25.json') as f:
        data = json.load(f)
    task = data['16644997']
    passed = 0
    for i, ex in enumerate(task['train']):
        pred = transform([row[:] for row in ex['input']])
        ok = pred == ex['output']
        print(f'Train {i}: {"PASS" if ok else "FAIL"}')
        if ok: passed += 1
    print(f'{passed}/{len(task["train"])} passing')
