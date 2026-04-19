from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    
    def find_all_bg(is_bg_func, total):
        return [i for i in range(total) if is_bg_func(i)]
    
    def merge_separators(seps, total):
        if not seps:
            return []
        bands = [[seps[0], seps[0]]]
        for s in seps[1:]:
            if s - bands[-1][1] <= 2:
                bands[-1][1] = s
            else:
                bands.append([s, s])
        return bands
    
    def get_groups(bands, total):
        groups = []
        prev_end = -1
        for start, end in bands:
            if start > prev_end + 1:
                groups.append((prev_end + 1, start - 1))
            prev_end = end
        if prev_end < total - 1:
            groups.append((prev_end + 1, total - 1))
        return groups
    
    sep_rows = find_all_bg(lambda r: all(grid[r][c] == bg for c in range(cols)), rows)
    sep_cols = find_all_bg(lambda c: all(grid[r][c] == bg for r in range(rows)), cols)
    
    row_bands = merge_separators(sep_rows, rows)
    col_bands = merge_separators(sep_cols, cols)
    
    row_groups = get_groups(row_bands, rows)
    col_groups = get_groups(col_bands, cols)
    
    result = []
    for r1, r2 in row_groups:
        row = []
        for c1, c2 in col_groups:
            colors = Counter()
            for r in range(r1, r2+1):
                for c in range(c1, c2+1):
                    if grid[r][c] != bg:
                        colors[grid[r][c]] += 1
            if colors:
                row.append(colors.most_common(1)[0][0])
            else:
                row.append(bg)
        result.append(row)
    return result
