from collections import Counter

def transform(input_grid):
    bg = Counter(v for r in input_grid for v in r).most_common(1)[0][0]
    R, C = len(input_grid), len(input_grid[0])
    g = input_grid

    bg_rows = {r for r in range(R) if all(g[r][c] == bg for c in range(C))}
    bg_cols = {c for c in range(C) if all(g[r][c] == bg for r in range(R))}
    nb_rows = sorted(set(range(R)) - bg_rows)
    nb_cols = sorted(set(range(C)) - bg_cols)

    def find_bands(indices):
        if not indices: return []
        bands, start = [], indices[0]
        for i in range(1, len(indices)):
            if indices[i] != indices[i-1] + 1:
                bands.append(list(range(start, indices[i-1]+1)))
                start = indices[i]
        bands.append(list(range(start, indices[-1]+1)))
        return bands

    row_bands = find_bands(nb_rows)
    col_bands = find_bands(nb_cols)

    def gravity(grid, bg_val, target_h):
        if not grid or not grid[0]: return grid
        Rg, Cg = len(grid), len(grid[0])
        col_vals = [[grid[r][c] for r in range(Rg) if grid[r][c] != bg_val] for c in range(Cg)]
        return [[col_vals[c][r] if r < len(col_vals[c]) else bg_val for c in range(Cg)] for r in range(target_h)]

    def compact(rows, cols):
        eff_rows = [r for r in rows if any(g[r][c] != bg for c in cols)]
        eff_cols = [c for c in cols if any(g[r][c] != bg for r in eff_rows)]
        if not eff_rows or not eff_cols:
            return [[bg]]
        
        sub_bands = find_bands(eff_cols)
        
        if len(sub_bands) <= 1:
            return [[g[r][c] for c in eff_cols] for r in eff_rows]
        
        sub_results = [compact(eff_rows, sb) for sb in sub_bands]
        target_h = max(len(sg) for sg in sub_results)
        
        for sg in sub_results:
            w = len(sg[0]) if sg else 0
            while len(sg) < target_h:
                sg.append([bg] * w)
        
        max_w = 0
        result = []
        for r in range(target_h):
            row_vals = []
            for sg in sub_results:
                band_row = list(sg[r])
                while band_row and band_row[0] == bg: band_row.pop(0)
                while band_row and band_row[-1] == bg: band_row.pop()
                row_vals.extend(band_row)
            max_w = max(max_w, len(row_vals))
            result.append(row_vals)
        
        for row in result:
            while len(row) < max_w: row.append(bg)
        
        return result

    # Process blocks
    blocks = {}
    for ri, rb in enumerate(row_bands):
        for ci, cb in enumerate(col_bands):
            blocks[(ri, ci)] = compact(rb, cb)

    # Tier heights and gravity
    tier_heights = {}
    for ri in range(len(row_bands)):
        heights = [len(blocks[(ri, ci)]) for ci in range(len(col_bands))]
        tier_heights[ri] = min(heights)

    for ri in range(len(row_bands)):
        th = tier_heights[ri]
        for ci in range(len(col_bands)):
            if len(blocks[(ri, ci)]) > th:
                blocks[(ri, ci)] = gravity(blocks[(ri, ci)], bg, th)

    # Determine if strip-concat or direct concat at top level
    # Use strip-concat when blocks have excess width (single row band case)
    use_strip = len(row_bands) == 1
    
    # Build output
    output = []
    for ri in range(len(row_bands)):
        th = tier_heights[ri]
        for r in range(th):
            row_vals = []
            for ci in range(len(col_bands)):
                bg_grid = blocks[(ri, ci)]
                if r < len(bg_grid):
                    band_row = list(bg_grid[r])
                    if use_strip:
                        while band_row and band_row[0] == bg: band_row.pop(0)
                        while band_row and band_row[-1] == bg: band_row.pop()
                    row_vals.extend(band_row)
                else:
                    w = len(bg_grid[0]) if bg_grid else 0
                    row_vals.extend([bg] * w)
            output.append(row_vals)

    max_w = max(len(r) for r in output) if output else 0
    for row in output:
        while len(row) < max_w: row.append(bg)

    return output
