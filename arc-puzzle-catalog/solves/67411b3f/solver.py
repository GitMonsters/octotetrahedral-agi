import json
from collections import Counter

def get_bg(grid):
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
    return Counter(flat).most_common(1)[0][0]

def get_mc_groups(grid, bg):
    R, C = len(grid), len(grid[0])
    visited = set()
    groups = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and (r,c) not in visited:
                comp = set()
                stack = [(r,c)]
                visited.add((r,c))
                while stack:
                    cr, cc = stack.pop()
                    comp.add((cr,cc))
                    for dr in [-1,0,1]:
                        for dc in [-1,0,1]:
                            if dr==0 and dc==0: continue
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc] != bg:
                                visited.add((nr,nc))
                                stack.append((nr,nc))
                colors = set(grid[rr][cc] for rr,cc in comp)
                if len(colors) >= 2:
                    groups.append(comp)
    return groups

def find_ref_color(grid, bg):
    mc_groups = get_mc_groups(grid, bg)
    if not mc_groups:
        return None
    color_counts = Counter()
    for grp in mc_groups:
        colors = set(grid[r][c] for r,c in grp)
        for c in colors:
            color_counts[c] += 1
    return color_counts.most_common(1)[0][0]

def find_template(grid, bg, ref_color):
    mc_groups = get_mc_groups(grid, bg)
    template = {}
    
    # From small MC groups (ref has 1 cell)
    for grp in mc_groups:
        ref_cells = [(r,c) for r,c in grp if grid[r][c] == ref_color]
        if len(ref_cells) != 1: continue
        ref_r, ref_c = ref_cells[0]
        for r, c in grp:
            dr, dc = r - ref_r, c - ref_c
            template[(dr, dc)] = 'R' if grid[r][c] == ref_color else 'G'
    
    # From large MC groups (ref has N×N cells)
    for grp in mc_groups:
        ref_cells = sorted((r,c) for r,c in grp if grid[r][c] == ref_color)
        if len(ref_cells) <= 1: continue
        ref_minr = min(r for r,c in ref_cells)
        ref_minc = min(c for r,c in ref_cells)
        ref_maxr = max(r for r,c in ref_cells)
        ref_maxc = max(c for r,c in ref_cells)
        N = max(ref_maxr - ref_minr + 1, ref_maxc - ref_minc + 1)
        if N <= 1: continue
        for r, c in grp:
            if grid[r][c] == ref_color: continue
            base_dr = (r - ref_minr) // N
            base_dc = (c - ref_minc) // N
            if (base_dr, base_dc) not in template:
                template[(base_dr, base_dc)] = 'G'
    
    template[(0, 0)] = 'R'
    return template

def pixel_scale_template(template, n):
    scaled = {}
    for (dr, dc), label in template.items():
        for i in range(n):
            for j in range(n):
                scaled[(dr*n + i, dc*n + j)] = label
    return scaled

def is_standalone(grid, bg, comp, ref_color):
    R, C = len(grid), len(grid[0])
    for cr, cc in comp:
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==0 and dc==0: continue
                nr, nc = cr+dr, cc+dc
                if 0<=nr<R and 0<=nc<C and grid[nr][nc] != bg and grid[nr][nc] != ref_color:
                    return False
    return True

def is_filled_rect(comp):
    if not comp: return True
    rs = [r for r,c in comp]
    cs = [c for r,c in comp]
    minr, maxr = min(rs), max(rs)
    minc, maxc = min(cs), max(cs)
    return len(comp) == (maxr - minr + 1) * (maxc - minc + 1)

def find_largest_block_size(cells):
    if not cells: return 1
    for n in [4, 3, 2]:
        for r, c in cells:
            if all((r+dr, c+dc) in cells for dr in range(n) for dc in range(n)):
                return n
    return 1

def best_template_alignment(cells, template_positions, N):
    scaled_positions = []
    for pos in template_positions:
        for i in range(N):
            for j in range(N):
                scaled_positions.append((pos[0]*N + i, pos[1]*N + j))
    
    best_score = -1
    best_anchor = None
    
    for ar, ac in cells:
        score = sum(1 for dr, dc in scaled_positions if (ar+dr, ac+dc) in cells)
        if score > best_score:
            best_score = score
            best_anchor = (ar, ac)
    
    return best_anchor, best_score, len(scaled_positions)

def solve(grid):
    R, C = len(grid), len(grid[0])
    bg = get_bg(grid)
    ref_color = find_ref_color(grid, bg)
    if ref_color is None:
        return [row[:] for row in grid]
    
    template = find_template(grid, bg, ref_color)
    mc_groups = get_mc_groups(grid, bg)
    result = [row[:] for row in grid]
    
    # Phase 1: Fill growing colors around ref cells in MC groups
    for grp in mc_groups:
        ref_cells = sorted((r,c) for r,c in grp if grid[r][c] == ref_color)
        grow_cells = set((r,c) for r,c in grp if grid[r][c] != ref_color)
        if not ref_cells or not grow_cells: continue
        
        ref_minr = min(r for r,c in ref_cells)
        ref_minc = min(c for r,c in ref_cells)
        ref_maxr = max(r for r,c in ref_cells)
        ref_maxc = max(c for r,c in ref_cells)
        N = max(ref_maxr - ref_minr + 1, ref_maxc - ref_minc + 1)
        
        scaled = pixel_scale_template(template, N)
        grow_color = Counter(grid[r][c] for r,c in grow_cells).most_common(1)[0][0]
        
        for (dr, dc), label in scaled.items():
            if label == 'G':
                pr = ref_minr + dr
                pc = ref_minc + dc
                if 0 <= pr < R and 0 <= pc < C and result[pr][pc] == bg:
                    result[pr][pc] = grow_color
    
    # Phase 2: Fill standalone ref-color components
    all_positions = list(template.keys())
    
    visited = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == ref_color and (r,c) not in visited:
                comp = set()
                stack = [(r,c)]
                visited.add((r,c))
                while stack:
                    cr, cc = stack.pop()
                    comp.add((cr,cc))
                    for dr in [-1,0,1]:
                        for dc in [-1,0,1]:
                            if dr==0 and dc==0: continue
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc]==ref_color:
                                visited.add((nr,nc))
                                stack.append((nr,nc))
                
                if len(comp) <= 1: continue
                if not is_standalone(grid, bg, comp, ref_color): continue
                if is_filled_rect(comp): continue
                
                N = find_largest_block_size(comp)
                anchor, score, total = best_template_alignment(comp, all_positions, N)
                
                if anchor is None: continue
                # Only fill if ALL existing cells match template positions (perfect alignment)
                if score != len(comp): continue
                # Only fill if there are missing cells
                if score >= total: continue
                
                scaled = pixel_scale_template(template, N)
                ar, ac = anchor
                for (dr, dc) in scaled:
                    pr = ar + dr
                    pc = ac + dc
                    if 0 <= pr < R and 0 <= pc < C and result[pr][pc] == bg:
                        result[pr][pc] = ref_color
    
    return result

if __name__ == '__main__':
    with open('/tmp/rearc45/67411b3f.json') as f:
        data = json.load(f)
    
    # Verify all training pairs
    all_pass = True
    for ti in range(len(data['train'])):
        inp = data['train'][ti]['input']
        out = data['train'][ti]['output']
        pred = solve(inp)
        RR, CC = len(inp), len(inp[0])
        diffs = sum(1 for r in range(RR) for c in range(CC) if pred[r][c] != out[r][c])
        status = "PASS" if diffs == 0 else "FAIL"
        print(f"Train {ti}: {status} ({diffs} diffs)")
        if diffs > 0: all_pass = False
    
    if all_pass:
        print("\nAll training pairs PASS!")
    
    # Generate test predictions
    for ti in range(len(data['test'])):
        inp = data['test'][ti]['input']
        pred = solve(inp)
        RR, CC = len(inp), len(inp[0])
        changes = sum(1 for r in range(RR) for c in range(CC) if pred[r][c] != inp[r][c])
        print(f"Test {ti}: {changes} changes")


def transform(grid): return solve(grid)
