from collections import Counter
from itertools import combinations

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    H = len(input_grid)
    W = len(input_grid[0])
    bg = Counter(v for row in input_grid for v in row).most_common(1)[0][0]
    
    pixels = [(r, c, input_grid[r][c]) for r in range(H) for c in range(W) if input_grid[r][c] != bg]
    if not pixels:
        return [[bg]*3 for _ in range(3)]
    
    input_counts = Counter(v for _,_,v in pixels)
    n = len(pixels)
    non_bg_vals = sorted(input_counts.keys())
    
    def divisors(nn):
        return [d for d in range(1, nn+1) if nn % d == 0]
    
    from itertools import product as iprod
    val_divs = {v: divisors(ic) for v, ic in input_counts.items()}
    count_combos = list(iprod(*(val_divs[v] for v in non_bg_vals)))
    
    cells9 = list(range(9))
    
    def place_values(vals_list, idx, used):
        if idx == len(vals_list):
            yield []
            return
        v, cnt = vals_list[idx]
        remaining = [c for c in cells9 if c not in used]
        for chosen in combinations(remaining, cnt):
            new_used = used | set(chosen)
            for rest in place_values(vals_list, idx+1, new_used):
                yield [(v, chosen)] + rest
    
    def min_stamps(G):
        """Compute minimum number of stamps of G needed to cover all pixels."""
        # For each pixel, find possible stamp positions
        px_stamps = []
        for pi, (r, c, v) in enumerate(pixels):
            possible = set()
            for pr in range(3):
                for pc in range(3):
                    if G[pr][pc] == v:
                        sr, sc = r-pr, c-pc
                        possible.add((sr, sc))
            px_stamps.append(possible)
        
        # For each stamp position, find which pixels it covers
        all_stamps = set()
        for ps in px_stamps:
            all_stamps |= ps
        
        stamp_coverage = {}
        for sr, sc in all_stamps:
            covered = set()
            for pi, (r, c, v) in enumerate(pixels):
                pr, pc = r-sr, c-sc
                if 0 <= pr <= 2 and 0 <= pc <= 2 and G[pr][pc] == v:
                    covered.add(pi)
            if covered:
                stamp_coverage[(sr,sc)] = frozenset(covered)
        
        # Greedy set cover
        uncovered = set(range(n))
        num_stamps = 0
        while uncovered:
            best_stamp = None
            best_cover = 0
            for pos, covered in stamp_coverage.items():
                ct = len(covered & uncovered)
                if ct > best_cover:
                    best_cover = ct
                    best_stamp = pos
            if best_stamp is None or best_cover == 0:
                return float('inf')
            uncovered -= stamp_coverage[best_stamp]
            num_stamps += 1
        return num_stamps
    
    best_grid = None
    best_key = (float('inf'), float('inf'))  # (stamps, non_bg)
    
    for combo in count_combos:
        total_non_bg = sum(combo)
        if total_non_bg > 9:
            continue
        
        val_counts = dict(zip(non_bg_vals, combo))
        vals_list = [(v, val_counts[v]) for v in non_bg_vals]
        
        for placement in place_values(vals_list, 0, set()):
            grid = [bg] * 9
            for v, positions in placement:
                for p in positions:
                    grid[p] = v
            
            G = [[grid[r*3+c] for c in range(3)] for r in range(3)]
            
            ns = min_stamps(G)
            key = (ns, total_non_bg)
            if key < best_key:
                best_key = key
                best_grid = G
    
    return best_grid if best_grid else [[bg]*3 for _ in range(3)]
