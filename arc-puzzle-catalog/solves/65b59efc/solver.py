import json, sys

def solve(grid):
    R, C = len(grid), len(grid[0])
    
    # Find separator columns from row 0 (⬜ = 5)
    sep_cols = sorted(c for c in range(C) if grid[0][c] == 5)
    
    # Column groups
    col_groups = []
    prev = -1
    for sc in sep_cols:
        if sc > prev + 1:
            col_groups.append(list(range(prev + 1, sc)))
        prev = sc
    if prev < C - 1:
        col_groups.append(list(range(prev + 1, C)))
    
    # Find separator rows (rows with > half ⬜ in the active grid area)
    active_C = (sep_cols[-1] + 1) if sep_cols else C
    sep_rows = sorted(r for r in range(R) if sum(1 for c in range(active_C) if grid[r][c] == 5) > active_C // 2)
    
    # Row groups
    row_groups = []
    prev = -1
    for sr in sep_rows:
        if sr > prev + 1:
            row_groups.append(list(range(prev + 1, sr)))
        prev = sr
    if prev < R - 1:
        row_groups.append(list(range(prev + 1, R)))
    
    template_rows = row_groups[0]
    pointer_rows = row_groups[1]
    dot_rows = row_groups[2]
    
    tile_h = len(template_rows)
    tile_w = len(col_groups[0])
    n_tiles = len(col_groups)
    
    # Extract templates: color -> pattern offsets
    templates = {}
    template_colors = []
    for j, cg in enumerate(col_groups):
        color = None
        pattern = []
        for ri, r in enumerate(template_rows):
            for ci, c in enumerate(cg):
                v = grid[r][c]
                if v != 0:
                    pattern.append((ri, ci))
                    if color is None:
                        color = v
        templates[color] = pattern
        template_colors.append(color)
    
    # Extract dots per column group
    dots = {}
    for j, cg in enumerate(col_groups):
        for r in dot_rows:
            for c in cg:
                v = grid[r][c]
                if v != 0:
                    dots[j] = v
                    break
            if j in dots:
                break
    
    # Map template color -> output color
    color_map = {tc: dots.get(j, 0) for j, tc in enumerate(template_colors)}
    
    # Parse pointer tiles: non-0 cells define output tile positions
    output_tiles = {}
    for j, cg in enumerate(col_groups):
        for ri, r in enumerate(pointer_rows):
            for ci, c in enumerate(cg):
                v = grid[r][c]
                if v != 0 and v in templates:
                    output_tiles[(ri, ci)] = (templates[v], color_map[v])
    
    # Build output: pointer tile is tile_h x tile_w, each position is a tile
    out = [[0] * (tile_w * tile_w) for _ in range(tile_h * tile_h)]
    for (ti, tj), (pattern, color) in output_tiles.items():
        br, bc = ti * tile_h, tj * tile_w
        for dr, dc in pattern:
            out[br + dr][bc + dc] = color
    
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
            if len(result) != len(exp) or len(result[0]) != len(exp[0]):
                print(f"  Size: {len(result)}x{len(result[0])} vs {len(exp)}x{len(exp[0])}")
            else:
                for r in range(len(result)):
                    for c in range(len(result[0])):
                        if result[r][c] != exp[r][c]:
                            print(f"  ({r},{c}): got {result[r][c]}, exp {exp[r][c]}")
