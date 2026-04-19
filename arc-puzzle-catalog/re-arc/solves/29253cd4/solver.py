from collections import Counter, defaultdict
import statistics

def find_center_1d(markers):
    """Find center of a 1D cross arm from marker positions."""
    if len(markers) == 1:
        return markers[0]
    markers = sorted(markers)
    # Find gaps
    gaps = [(markers[i+1] - markers[i], i) for i in range(len(markers)-1)]
    if not gaps:
        return markers[0]
    max_gap, max_idx = max(gaps)
    other_gaps = [g for g, _ in gaps if _ != max_idx]
    second_max = max(other_gaps) if other_gaps else 0
    
    if max_gap > 1 and (second_max == 0 or max_gap > second_max * 1.5):
        # Clear dominant gap - center is in the gap
        return (markers[max_idx] + markers[max_idx+1]) // 2
    else:
        # No clear gap - use median
        return int(statistics.median(markers))

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    counts = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    all_colors = sorted(counts.items(), key=lambda x: -x[1])
    
    if len(all_colors) < 2:
        return output
    
    marker_color = all_colors[-1][0]
    marker_positions = set((r,c) for r in range(rows) for c in range(cols) if grid[r][c] == marker_color)
    
    if not marker_positions:
        return output
    
    # Cluster markers
    remaining = set(marker_positions)
    clusters = []
    while remaining:
        start = next(iter(remaining))
        cluster = set()
        queue = [start]
        cluster.add(start)
        remaining.discard(start)
        while queue:
            cr, cc = queue.pop(0)
            for mr, mc in list(remaining):
                if abs(mr-cr) + abs(mc-cc) <= 4:
                    cluster.add((mr, mc))
                    remaining.discard((mr, mc))
                    queue.append((mr, mc))
        clusters.append(list(cluster))
    
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        
        row_groups = defaultdict(list)
        col_groups = defaultdict(list)
        for r, c in cluster:
            row_groups[r].append(c)
            col_groups[c].append(r)
        
        max_row_count = max(len(v) for v in row_groups.values())
        max_col_count = max(len(v) for v in col_groups.values())
        
        best_rows = [r for r, v in row_groups.items() if len(v) == max_row_count]
        best_cols = [c for c, v in col_groups.items() if len(v) == max_col_count]
        
        if max_row_count >= 2 and max_col_count >= 2:
            center_row = best_rows[0] if len(best_rows) == 1 else int(statistics.median(best_rows))
            center_col = best_cols[0] if len(best_cols) == 1 else int(statistics.median(best_cols))
        elif max_row_count >= 2:
            center_row = best_rows[0] if len(best_rows) == 1 else int(statistics.median(best_rows))
            row_cols = sorted(row_groups[center_row])
            # Check for vertical arm col
            vert_col = None
            for c, rs in col_groups.items():
                if any(r != center_row for r in rs):
                    vert_col = c
                    break
            if vert_col is not None:
                center_col = vert_col
            else:
                center_col = find_center_1d(row_cols)
        elif max_col_count >= 2:
            center_col = best_cols[0] if len(best_cols) == 1 else int(statistics.median(best_cols))
            col_rows = sorted(col_groups[center_col])
            # Check for horizontal arm row
            horiz_row = None
            for r, cs in row_groups.items():
                if any(c != center_col for c in cs):
                    horiz_row = r
                    break
            if horiz_row is not None:
                center_row = horiz_row
            else:
                center_row = find_center_1d(col_rows)
        else:
            all_rs = sorted(set(r for r, c in cluster))
            all_cs = sorted(set(c for r, c in cluster))
            center_row = find_center_1d(all_rs)
            center_col = find_center_1d(all_cs)
        
        # Arm length
        arm_length = 0
        for r, c in cluster:
            if r == center_row:
                arm_length = max(arm_length, abs(c - center_col))
            if c == center_col:
                arm_length = max(arm_length, abs(r - center_row))
        if arm_length == 0:
            for r, c in cluster:
                arm_length = max(arm_length, abs(r - center_row), abs(c - center_col))
        
        # Fill cross
        for d in range(-arm_length, arm_length + 1):
            nr, nc = center_row, center_col + d
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in marker_positions:
                output[nr][nc] = 5
            nr, nc = center_row + d, center_col
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in marker_positions:
                output[nr][nc] = 5
    
    return output

