#!/usr/bin/env python3
"""
Solver for ARC-AGI task 414297c0.

Pattern:
1. Find the large filled rectangle (dense rectangular region)
2. Extract the filled rectangle as output
3. For each special value pair (inside/outside rect):
   - Find the rectangular region around outside value that contains nearby 2s
   - Place that pattern at the target position in output
"""

def solve(grid: list[list[int]]) -> list[list[int]]:
    h, w = len(grid), len(grid[0])
    
    # Find the filled rectangle
    color_info = {}
    for color in range(1, 10):
        cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == color]
        if len(cells) > 10:
            min_r = min(r for r, c in cells)
            max_r = max(r for r, c in cells)
            min_c = min(c for r, c in cells)
            max_c = max(c for r, c in cells)
            area = (max_r - min_r + 1) * (max_c - min_c + 1)
            density = len(cells) / area
            if density > 0.7:
                color_info[color] = {
                    'bounds': (min_r, max_r, min_c, max_c),
                    'cells': set(cells),
                    'density': density
                }
    
    if not color_info:
        return grid
    
    # Find the rectangle with highest density
    fill_color = max(color_info, key=lambda c: color_info[c]['density'])
    min_r, max_r, min_c, max_c = color_info[fill_color]['bounds']
    
    out_h = max_r - min_r + 1
    out_w = max_c - min_c + 1
    output = [[fill_color] * out_w for _ in range(out_h)]
    
    # Find all special values (their locations and patterns)
    special_values = {}
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v not in [0, fill_color, 2]:
                if v not in special_values:
                    special_values[v] = []
                special_values[v].append((r, c))
    
    # For each special value, one location is inside rect, one outside
    for value, locations in special_values.items():
        inside_loc = None
        outside_loc = None
        
        for loc_r, loc_c in locations:
            if min_r <= loc_r <= max_r and min_c <= loc_c <= max_c:
                inside_loc = (loc_r, loc_c)
            else:
                outside_loc = (loc_r, loc_c)
        
        if inside_loc is None or outside_loc is None:
            continue
        
        # Target output position
        out_r = inside_loc[0] - min_r
        out_c = inside_loc[1] - min_c
        
        # Extract pattern around outside_loc
        # Include the value and all 2s within a reasonable rectangular region
        
        # Start with just the value
        pattern_cells = {outside_loc: value}
        visited = set()
        queue = [outside_loc]
        visited.add(outside_loc)
        
        # BFS to find connected 2s and the value
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    v = grid[nr][nc]
                    if v in [2, value]:
                        visited.add((nr, nc))
                        pattern_cells[(nr, nc)] = v
                        queue.append((nr, nc))
        
        # Also include isolated 2s within a small range of the pattern
        pr = [rr for rr, cc in pattern_cells.keys()]
        pc = [cc for rr, cc in pattern_cells.keys()]
        p_min_r, p_max_r = min(pr), max(pr)
        p_min_c, p_max_c = min(pc), max(pc)
        
        # Expand search box by 1 in each direction to catch isolated 2s
        for r in range(max(0, p_min_r - 1), min(h, p_max_r + 2)):
            for c in range(max(0, p_min_c - 1), min(w, p_max_c + 2)):
                if (r, c) not in pattern_cells and grid[r][c] == 2:
                    pattern_cells[(r, c)] = 2
        
        # Get pattern bounds
        pr = [rr for rr, cc in pattern_cells.keys()]
        pc = [cc for rr, cc in pattern_cells.keys()]
        if not pr or not pc:
            continue
        p_min_r, p_max_r = min(pr), max(pr)
        p_min_c, p_max_c = min(pc), max(pc)
        
        # Calculate offset from pattern min to the special value
        offset_r = outside_loc[0] - p_min_r
        offset_c = outside_loc[1] - p_min_c
        
        # Place pattern in output with value at (out_r, out_c)
        pattern_top_r = out_r - offset_r
        pattern_top_c = out_c - offset_c
        
        for (pr, pc), pv in pattern_cells.items():
            out_pr = pattern_top_r + (pr - p_min_r)
            out_pc = pattern_top_c + (pc - p_min_c)
            if 0 <= out_pr < out_h and 0 <= out_pc < out_w:
                output[out_pr][out_pc] = pv
    
    return output


def main():
    import json
    import sys
    
    path = sys.argv[1] if len(sys.argv) > 1 else '~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/414297c0.json'
    path = path.replace('~', '/Users/evanpieser')
    
    with open(path) as f:
        data = json.load(f)
    
    print(f"Testing {path}")
    passed = 0
    for i, example in enumerate(data['train']):
        result = solve(example['input'])
        expected = example['output']
        
        match = result == expected
        status = "PASS" if match else "FAIL"
        print(f"  Example {i}: {status}")
        
        if not match:
            print(f"    Expected shape: {len(expected)}x{len(expected[0]) if expected else 0}")
            print(f"    Got shape: {len(result)}x{len(result[0]) if result else 0}")
        
        if match:
            passed += 1
    
    print(f"\nResult: {passed}/{len(data['train'])} training examples passed")
    return 0 if passed == len(data['train']) else 1


if __name__ == '__main__':
    main()
