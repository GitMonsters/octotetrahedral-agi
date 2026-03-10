import json, sys
from collections import Counter, deque

def solve(grid):
    R, C = len(grid), len(grid[0])
    
    # Background = most common color
    counts = Counter()
    for r in range(R):
        for c in range(C):
            counts[grid[r][c]] += 1
    bg = counts.most_common(1)[0][0]
    
    # Find connected components of non-bg cells (8-connectivity)
    visited = [[False]*C for _ in range(R)]
    components = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] != bg:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                components.append(comp)
    
    # Take 2 largest components
    components.sort(key=len, reverse=True)
    components = components[:2]
    
    # Determine selector (1 non-bg color) vs tile (2+ non-bg colors)
    def get_nonbg_colors(comp):
        return set(grid[r][c] for r, c in comp) - {bg}
    
    c0, c1 = get_nonbg_colors(components[0]), get_nonbg_colors(components[1])
    if len(c0) <= 1 and len(c1) > 1:
        sel_comp, tile_comp = components[0], components[1]
    elif len(c1) <= 1 and len(c0) > 1:
        sel_comp, tile_comp = components[1], components[0]
    else:
        sel_comp, tile_comp = (components[0], components[1]) if len(components[0]) <= len(components[1]) else (components[1], components[0])
    
    sel_color = list(get_nonbg_colors(sel_comp))[0]
    
    # Bounding boxes
    def bbox(comp):
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        return min(rs), min(cs), max(rs)+1, max(cs)+1
    
    sr1, sc1, sr2, sc2 = bbox(sel_comp)
    tr1, tc1, tr2, tc2 = bbox(tile_comp)
    sel_h, sel_w = sr2-sr1, sc2-sc1
    tile_h, tile_w = tr2-tr1, tc2-tc1
    
    # Extract selector pattern
    sel = [[grid[r][c] == sel_color for c in range(sc1, sc2)] for r in range(sr1, sr2)]
    
    # Extract tile pattern
    tile = [[grid[r][c] for c in range(tc1, tc2)] for r in range(tr1, tr2)]
    
    # Build output
    out = [[bg]*(sel_w*tile_w) for _ in range(sel_h*tile_h)]
    for si in range(sel_h):
        for sj in range(sel_w):
            if sel[si][sj]:
                for ti in range(tile_h):
                    for tj in range(tile_w):
                        out[si*tile_h+ti][sj*tile_w+tj] = tile[ti][tj]
    return out

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        ok = result == ex['output']
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            exp = ex['output']
            if len(result) != len(exp) or (result and len(result[0]) != len(exp[0])):
                print(f"  Size: {len(result)}x{len(result[0]) if result else 0} vs {len(exp)}x{len(exp[0])}")
            else:
                diffs = 0
                for r in range(len(result)):
                    for c in range(len(result[0])):
                        if result[r][c] != exp[r][c]:
                            if diffs < 10: print(f"  ({r},{c}): got {result[r][c]}, exp {exp[r][c]}")
                            diffs += 1
                if diffs > 10: print(f"  ... {diffs} diffs total")
