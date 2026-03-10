import json
from collections import Counter, deque

def solve(grid):
    R, C = len(grid), len(grid[0])
    
    # 1. Find ⬜ frame
    frame_cells = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 5:
                frame_cells.add((r, c))
    
    fr_min = min(r for r,c in frame_cells)
    fr_max = max(r for r,c in frame_cells)
    fc_min = min(c for r,c in frame_cells)
    fc_max = max(c for r,c in frame_cells)
    int_r_start, int_r_end = fr_min + 1, fr_max - 1
    int_c_start, int_c_end = fc_min + 1, fc_max - 1
    
    # 2. Background color (most common non-⬜)
    color_counts = Counter()
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 5:
                color_counts[grid[r][c]] += 1
    bg = color_counts.most_common(1)[0][0]
    
    # 3. Find non-bg cells outside the frame
    outside_cells = {}
    for r in range(R):
        for c in range(C):
            if (r, c) in frame_cells:
                continue
            if int_r_start <= r <= int_r_end and int_c_start <= c <= int_c_end:
                continue
            if grid[r][c] != bg and grid[r][c] != 5:
                outside_cells[(r, c)] = grid[r][c]
    
    # 4. Group into connected components (4-connectivity)
    visited = set()
    components = []
    for cell in outside_cells:
        if cell in visited:
            continue
        comp = []
        queue = deque([cell])
        visited.add(cell)
        while queue:
            r, c = queue.popleft()
            comp.append((r, c, outside_cells[(r,c)]))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (nr, nc) in outside_cells and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        components.append(comp)
    
    # 5. Classify: 2-color = key pair, 1-color = template
    key_pairs = {}   # left_color -> right_color
    template_cells = {}  # template_color -> set of (r,c)
    
    for comp in components:
        colors = set(clr for _, _, clr in comp)
        if len(colors) == 2:
            min_col = min(c for _, c, _ in comp)
            left_color = next(clr for _, c, clr in comp if c == min_col)
            right_color = (colors - {left_color}).pop()
            key_pairs[left_color] = right_color
        elif len(colors) == 1:
            template_color = list(colors)[0]
            if template_color not in template_cells:
                template_cells[template_color] = set()
            for r2, c2, _ in comp:
                template_cells[template_color].add((r2, c2))
    
    # Build template patterns from merged cells
    templates = {}
    for template_color, cell_set in template_cells.items():
        rs = [r for r, c in cell_set]
        cs = [c for r, c in cell_set]
        min_r, max_r = min(rs), max(rs)
        min_c2, max_c = min(cs), max(cs)
        pattern = [[((r,c) in cell_set) for c in range(min_c2, max_c+1)] for r in range(min_r, max_r+1)]
        templates[template_color] = pattern
    
    # 6. Parse grid blocks inside frame
    row_seps = [r for r in range(int_r_start, int_r_end+1)
                if all(grid[r][c] == bg for c in range(int_c_start, int_c_end+1))]
    col_seps = [c for c in range(int_c_start, int_c_end+1)
                if all(grid[r][c] == bg for r in range(int_r_start, int_r_end+1))]
    
    def get_regions(seps, start, end):
        regions = []
        prev = start - 1
        for s in sorted(seps):
            if s > prev + 1:
                regions.append((prev + 1, s - 1))
            prev = s
        if prev < end:
            regions.append((prev + 1, end))
        return regions
    
    row_regions = get_regions(row_seps, int_r_start, int_r_end)
    col_regions = get_regions(col_seps, int_c_start, int_c_end)
    
    # 7. Build output
    out_h = fr_max - fr_min + 1
    out_w = fc_max - fc_min + 1
    output = [[bg]*out_w for _ in range(out_h)]
    
    for r, c in frame_cells:
        output[r - fr_min][c - fc_min] = 5
    
    for r_start, r_end in row_regions:
        for c_start, c_end in col_regions:
            block_colors = Counter()
            for r in range(r_start, r_end+1):
                for c in range(c_start, c_end+1):
                    if grid[r][c] != bg:
                        block_colors[grid[r][c]] += 1
            if not block_colors:
                continue
            block_color = block_colors.most_common(1)[0][0]
            if block_color not in templates or block_color not in key_pairs:
                continue
            pattern = templates[block_color]
            out_color = key_pairs[block_color]
            bh = r_end - r_start + 1
            bw = c_end - c_start + 1
            for dr in range(bh):
                for dc in range(bw):
                    if dr < len(pattern) and dc < len(pattern[0]) and pattern[dr][dc]:
                        output[r_start - fr_min + dr][c_start - fc_min + dc] = out_color
    
    return output

# Test
DIR = '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/'
with open(f'{DIR}edb79dae.json') as f:
    task = json.load(f)

for i, ex in enumerate(task['train']):
    result = solve(ex['input'])
    if result == ex['output']:
        print(f"Train {i}: PASS")
    else:
        print(f"Train {i}: FAIL")
        for r in range(len(result)):
            for c in range(len(result[0])):
                if r < len(ex['output']) and c < len(ex['output'][0]):
                    if result[r][c] != ex['output'][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]} expected {ex['output'][r][c]}")
        if len(result) != len(ex['output']) or (result and len(result[0]) != len(ex['output'][0])):
            print(f"  Size mismatch: got {len(result)}x{len(result[0]) if result else 0} expected {len(ex['output'])}x{len(ex['output'][0])}")
