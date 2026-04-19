def transform(grid):
    from collections import Counter, defaultdict

    rows, cols = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    sep = Counter(flat).most_common(1)[0][0]
    non_sep = [v for v in flat if v != sep]
    fill = Counter(non_sep).most_common(1)[0][0]

    visited = [[False]*cols for _ in range(rows)]
    panels = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != sep and not visited[r][c]:
                stack = [(r, c)]
                visited[r][c] = True
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] != sep:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                minr = min(r for r,_ in cells)
                maxr = max(r for r,_ in cells)
                minc = min(c for _,c in cells)
                maxc = max(c for _,c in cells)
                panels.append((minr, minc, maxr, maxc))

    sep_rows = set(r for r in range(rows) if all(grid[r][c] == sep for c in range(cols)))

    def find_row_group(r):
        start = r
        while start > 0 and start - 1 not in sep_rows:
            start -= 1
        return start

    row_group_map = defaultdict(list)
    for p in panels:
        row_group_map[find_row_group(p[0])].append(p)
    groups = [row_group_map[k] for k in sorted(row_group_map.keys())]

    col_widths = None
    for g in groups:
        if len(g) == 3:
            col_widths = sorted(p[3]-p[1]+1 for p in g)
            break

    def get_content(r1, c1, r2, c2):
        return [grid[r][c1:c2+1] for r in range(r1, r2+1)]

    def trim_sep_rows(content, sep_val):
        while content and all(c == sep_val for c in content[0]):
            content = content[1:]
        while content and all(c == sep_val for c in content[-1]):
            content = content[:-1]
        return content

    processed = []
    for g in groups:
        g_sorted = sorted(g, key=lambda p: p[1])
        if len(g_sorted) == 3:
            processed.append([(p[3]-p[1]+1, get_content(*p)) for p in g_sorted])
        elif len(g_sorted) == 2:
            widths = [p[3]-p[1]+1 for p in g_sorted]
            merged_idx = None
            other_w = None
            for i, w in enumerate(widths):
                if w in col_widths:
                    other_w = w
                else:
                    merged_idx = i
            if merged_idx is None:
                processed.append([(w, get_content(*p)) for w, p in zip(widths, g_sorted)])
                continue
            merged_p = g_sorted[merged_idx]
            other_p = g_sorted[1 - merged_idx]
            remaining = sorted([w for w in col_widths if w != other_w])
            w1, w2 = remaining[0], remaining[1]
            content = get_content(*merged_p)
            other_content = get_content(*other_p)
            target_h = len(other_content)

            for wa, wb in [(w1, w2), (w2, w1)]:
                left = [row[:wa] for row in content]
                right = [row[wa:] for row in content]
                lt = trim_sep_rows(left, sep)
                rt = trim_sep_rows(right, sep)
                if len(lt) == target_h and len(rt) == target_h:
                    if merged_idx == 0:
                        processed.append([(wa, lt), (wb, rt), (other_w, other_content)])
                    else:
                        processed.append([(other_w, other_content), (wa, lt), (wb, rt)])
                    break

    def has_dec(content, side):
        h, w = len(content), len(content[0])
        if side == 'left':
            return any(content[r][0] not in (fill, sep) for r in range(h))
        elif side == 'right':
            return any(content[r][w-1] not in (fill, sep) for r in range(h))
        elif side == 'top':
            return any(content[0][c] not in (fill, sep) for c in range(w))
        elif side == 'bottom':
            return any(content[h-1][c] not in (fill, sep) for c in range(w))

    output_sections = []
    for pg in processed:
        left_p = center_p = right_p = None
        section = 'middle'
        for w, content in pg:
            ld = has_dec(content, 'left')
            rd = has_dec(content, 'right')
            td = has_dec(content, 'top')
            bd = has_dec(content, 'bottom')
            if ld and not rd:
                left_p = content
            elif rd and not ld:
                right_p = content
            else:
                center_p = content
                if td and not bd:
                    section = 'top'
                elif bd and not td:
                    section = 'bottom'
        contents = [c for _, c in pg]
        if left_p is None: left_p = contents[0]
        if center_p is None: center_p = contents[1] if len(contents) > 1 else contents[0]
        if right_p is None: right_p = contents[-1]
        output_sections.append((section, left_p, center_p, right_p))

    order_map = {'top': 0, 'middle': 1, 'bottom': 2}
    output_sections.sort(key=lambda x: order_map[x[0]])

    result = []
    for _, left, center, right in output_sections:
        for r in range(len(left)):
            result.append(left[r] + center[r] + right[r])
    return result
