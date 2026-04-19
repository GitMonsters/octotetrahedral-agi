from math import gcd
from functools import reduce
from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background (most common) and shape colors
    vals = Counter()
    for row in grid:
        vals.update(row)
    bg = vals.most_common(1)[0][0]
    
    # Create binary mask
    mask = [[1 if grid[r][c] != bg else 0 for c in range(cols)] for r in range(rows)]
    
    # Find block size from run lengths
    run_lengths = []
    for r in range(rows):
        run = 0
        for c in range(cols):
            if mask[r][c]:
                run += 1
            else:
                if run > 0:
                    run_lengths.append(run)
                run = 0
        if run > 0:
            run_lengths.append(run)
    for c in range(cols):
        run = 0
        for r in range(rows):
            if mask[r][c]:
                run += 1
            else:
                if run > 0:
                    run_lengths.append(run)
                run = 0
        if run > 0:
            run_lengths.append(run)
    
    bs = reduce(gcd, run_lengths) if run_lengths else 1
    
    # Find shape bounding box
    shape_coords = [(r, c) for r in range(rows) for c in range(cols) if mask[r][c]]
    r_min = min(r for r, c in shape_coords)
    c_min = min(c for r, c in shape_coords)
    
    # Create block grid aligned to shape origin
    blocks = set()
    for r, c in shape_coords:
        blocks.add(((r - r_min) // bs, (c - c_min) // bs))
    
    max_br = max(b[0] for b in blocks) + 1
    max_bc = max(b[1] for b in blocks) + 1
    
    block_grid = tuple(tuple(1 if (br, bc) in blocks else 0 for bc in range(max_bc)) for br in range(max_br))
    
    # Generate all 8 rotations/reflections and find canonical form
    def rotate90(g):
        r, c = len(g), len(g[0])
        return tuple(tuple(g[r-1-j][i] for j in range(r)) for i in range(c))
    
    def reflect(g):
        return tuple(tuple(reversed(row)) for row in g)
    
    def grid_to_key(g):
        return g
    
    variants = set()
    g = block_grid
    for _ in range(4):
        variants.add(grid_to_key(g))
        variants.add(grid_to_key(reflect(g)))
        g = rotate90(g)
    
    # Known patterns
    plus_pattern = ((0,1,0),(1,1,1),(0,1,0))
    x_pattern = ((1,0,1),(0,1,0),(1,0,1))
    diag_pattern = ((1,1,0),(1,0,1),(0,1,0))
    l_pattern = ((0,1,1),(0,1,1),(1,0,0))
    
    def get_all_variants(pat):
        vs = set()
        g = pat
        for _ in range(4):
            vs.add(g)
            vs.add(reflect(g))
            g = rotate90(g)
        return vs
    
    plus_variants = get_all_variants(plus_pattern)
    x_variants = get_all_variants(x_pattern)
    diag_variants = get_all_variants(diag_pattern)
    l_variants = get_all_variants(l_pattern)
    
    if variants & plus_variants:
        return [[0]]
    elif variants & x_variants:
        return [[1]]
    elif variants & diag_variants:
        return [[3]]
    elif variants & l_variants:
        return [[5]]
    else:
        return [[0]]
