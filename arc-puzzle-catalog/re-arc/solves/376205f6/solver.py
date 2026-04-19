from typing import List
from collections import Counter, deque


def transform(grid: List[List[int]]) -> List[List[int]]:
    H, W = len(grid), len(grid[0])
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    non_bg_colors = set(flat) - {bg}

    adj_colors = {c: set() for c in non_bg_colors}
    color_counts = Counter(flat)
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v == bg:
                continue
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    nv = grid[nr][nc]
                    if nv != bg and nv != v:
                        adj_colors[v].add(nv)
    connector = max(non_bg_colors,
                    key=lambda c: (len(adj_colors[c]), -color_counts[c]))

    visited = [[False] * W for _ in range(H)]
    components = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < H and 0 <= nc < W
                                and not visited[nr][nc]
                                and grid[nr][nc] != bg):
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(comp)
    components.sort(key=lambda g: min(r for r, c in g))

    shape_groups = []
    isolated_groups = []
    for comp in components:
        conn_cells = sorted([(r, c) for r, c in comp if grid[r][c] == connector])
        shape_cells = [(r, c) for r, c in comp if grid[r][c] != connector]
        if not shape_cells:
            isolated_groups.append({
                'cells': comp,
                'min_row': min(r for r, c in comp),
                'max_row': max(r for r, c in comp),
            })
            continue
        shape_color = Counter(grid[r][c] for r, c in shape_cells).most_common(1)[0][0]
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        if not conn_cells:
            entry, exit_ = None, None
        elif len(conn_cells) == 1:
            rc = conn_cells[0]
            # connector at top of group -> entry (last in chain)
            # connector at bottom of group -> exit (first in chain)
            if rc[0] == min_r:
                entry, exit_ = rc, None
            else:
                entry, exit_ = None, rc
        else:
            entry = min(conn_cells, key=lambda x: x[0])
            exit_ = max(conn_cells, key=lambda x: x[0])
        shape_groups.append({
            'cells': comp, 'conn_cells': conn_cells, 'shape_color': shape_color,
            'entry': entry, 'exit': exit_,
            'min_row': min_r, 'max_row': max_r, 'shift': 0,
        })

    # Column shifts: work backward from last group (shift=0)
    n = len(shape_groups)
    for i in range(n - 2, -1, -1):
        curr, nxt = shape_groups[i], shape_groups[i + 1]
        exit_col = (curr['exit'][1] if curr['exit']
                    else (curr['conn_cells'][-1][1] if curr['conn_cells'] else 0))
        entry_col = (nxt['entry'][1] if nxt['entry']
                     else (nxt['conn_cells'][0][1] if nxt['conn_cells'] else 0))
        curr['shift'] = (entry_col + nxt['shift']) - exit_col

    # Isolated connector span: empty rows preserved between shape groups
    iso_span_rows: set = set()
    if isolated_groups:
        iso_min = min(g['min_row'] for g in isolated_groups)
        iso_max = max(g['max_row'] for g in isolated_groups)
        iso_span_rows = set(range(iso_min, iso_max + 1))
        if len(shape_groups) == 2:
            sg1, sg2 = shape_groups[0], shape_groups[1]
            ex_r = sg1['exit'][0] if sg1['exit'] else sg1['max_row']
            ex_c = sg1['exit'][1] if sg1['exit'] else 0
            en_r = sg2['entry'][0] if sg2['entry'] else sg2['min_row']
            en_c = sg2['entry'][1] if sg2['entry'] else 0
            sg2['shift'] = 0
            sg1_height = sg1['max_row'] - sg1['min_row'] + 1
            iso_span = iso_max - iso_min + 1
            # Output row positions: sg1 exit at row (sg1_height-1),
            # sg2 entry at row (sg1_height + iso_span).
            row_diff_out = iso_span + 1   # (sg1_height+iso_span) - (sg1_height-1)
            row_diff_in = en_r - ex_r
            col_diff_in = en_c - ex_c
            if row_diff_in != 0:
                slope = col_diff_in / row_diff_in
                sg1['shift'] = round(en_c - slope * row_diff_out) - ex_c

    row_to_group = {r: sg for sg in shape_groups for r, _ in sg['cells']}
    all_nb = sorted(r for r in range(H) if any(grid[r][c] != bg for c in range(W)))
    if not isolated_groups:
        output_rows = [r for r in all_nb if r in row_to_group]
    else:
        shape_rows = set(r for sg in shape_groups for r, _ in sg['cells'])
        output_rows = sorted(shape_rows | iso_span_rows)

    result = [[bg] * W for _ in range(len(output_rows))]
    for out_r, in_r in enumerate(output_rows):
        sg = row_to_group.get(in_r)
        if sg is None:
            continue
        shift = sg['shift']
        sc = sg['shape_color']
        for c in range(W):
            v = grid[in_r][c]
            if v == bg:
                continue
            new_c = c + shift
            if 0 <= new_c < W:
                result[out_r][new_c] = sc if v == connector else v
    return result
