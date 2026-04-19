from collections import Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    result = [[bg]*cols for _ in range(rows)]
    
    lines = {}
    isolated = {}
    for r in range(rows):
        non_bg = [(c, grid[r][c]) for c in range(cols) if grid[r][c] != bg]
        if not non_bg: continue
        colors = set(v for _,v in non_bg)
        if len(colors) == 1:
            color = list(colors)[0]
            cols_list = sorted(c for c,_ in non_bg)
            if cols_list == list(range(cols_list[0], cols_list[-1]+1)):
                if len(cols_list) == 1:
                    isolated[r] = (cols_list[0], color)
                else:
                    lines[r] = (cols_list[0], cols_list[-1], color)
    
    zero_lines = {r: (cs,ce) for r,(cs,ce,c) in lines.items() if c == 0}
    nine_lines = {r: (cs,ce) for r,(cs,ce,c) in lines.items() if c == 9}
    other_lines = {r: (cs,ce,c) for r,(cs,ce,c) in lines.items() if c not in (0,9)}
    
    nine_specs = set(nine_lines.values())
    zero_matching_nine = {r: (cs,ce) for r,(cs,ce) in zero_lines.items() if (cs,ce) in nine_specs}
    zero_not_matching = {r: (cs,ce) for r,(cs,ce) in zero_lines.items() if (cs,ce) not in nine_specs}
    
    nine_by_spec = {}
    for r,(cs,ce) in nine_lines.items():
        nine_by_spec.setdefault((cs,ce), []).append(r)
    
    disappeared_rows = set()
    surviving_nine = {}
    for spec, nine_rows in nine_by_spec.items():
        if len(nine_rows) >= 2:
            for r in nine_rows: disappeared_rows.add(r)
        else:
            surviving_nine[nine_rows[0]] = spec
    
    for r_8, (cs,ce,col) in list(other_lines.items()):
        adj_to_9 = any(abs(r_8 - r9) <= 1 for r9 in nine_lines)
        matching_zeros = [r0 for r0,(cs0,ce0) in zero_lines.items() if cs0==cs and ce0==ce]
        if adj_to_9 and matching_zeros:
            disappeared_rows.add(r_8)
            for r0 in matching_zeros: disappeared_rows.add(r0)
    
    gap_start = min(disappeared_rows) if disappeared_rows else -1
    gap_end = max(disappeared_rows) if disappeared_rows else -1
    
    converted_from_8 = {}
    for r,(cs,ce,col) in other_lines.items():
        if r not in disappeared_rows:
            for c in range(cs, ce+1): result[r][c] = 0
            converted_from_8[r] = (cs,ce)
    
    for r,(cs,ce) in zero_not_matching.items():
        if r not in disappeared_rows:
            for c in range(cs, ce+1): result[r][c] = 0
    
    for r,(cs,ce) in zero_matching_nine.items():
        if r not in disappeared_rows:
            for c in range(cs, ce+1): result[r][c] = 9
    
    for r,(cs,ce) in surviving_nine.items():
        for c in range(cs, ce+1): result[r][c] = 9
    
    for r,(c,col) in isolated.items():
        result[r][c] = 9 if col == 8 else col
    
    surviving_nine_spec = None
    for r,(cs,ce) in surviving_nine.items():
        surviving_nine_spec = (cs,ce,9); break
    
    if surviving_nine_spec is None:
        spec_count = Counter()
        for r,(cs,ce) in zero_not_matching.items():
            if r not in disappeared_rows: spec_count[(cs,ce)] += 1
        for r,(cs,ce,col) in other_lines.items():
            if r not in disappeared_rows: spec_count[(cs,ce)] += 1
        if not spec_count: return result
        best = spec_count.most_common(1)[0][0]
        template_spec = (best[0], best[1], 0)
    else:
        template_spec = surviving_nine_spec
    
    def add_line(rg, row, cs, ce, color):
        if 0 <= row < rows and all(rg[row][c] == bg for c in range(cs, ce+1)):
            for c in range(cs, ce+1): rg[row][c] = color
    
    ts_cs, ts_ce, ts_color = template_spec
    
    if gap_start >= 0:
        for r,(cs,ce) in zero_not_matching.items():
            if r not in disappeared_rows and gap_start <= r <= gap_end:
                add_line(result, r+1, ts_cs, ts_ce, ts_color)
        for r,(cs,ce) in converted_from_8.items():
            if gap_start <= r <= gap_end:
                add_line(result, r+1, ts_cs, ts_ce, ts_color)
        for r,(cs,ce) in converted_from_8.items():
            if not (gap_start <= r <= gap_end):
                if not any(abs(r-d) <= 1 for d in disappeared_rows):
                    add_line(result, (r+1 if r < gap_start else r-1), ts_cs, ts_ce, ts_color)
        for r,(c,col) in isolated.items():
            if col == 9 and gap_start <= r <= gap_end:
                add_line(result, r-1, ts_cs, ts_ce, ts_color)
    
    return result
