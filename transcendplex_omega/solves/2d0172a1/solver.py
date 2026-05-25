import json, sys
from collections import Counter, deque
import heapq


def ring_1d(seq, fg_val):
    result = []; count = 0; in_fg = False
    for v in seq:
        if v == fg_val:
            if not in_fg: count += 1; in_fg = True
            result.append(2 * count - 1)
        else:
            in_fg = False; result.append(2 * count)
    return result


def compute_ring_grid(grid, fg, rows, cols):
    left = [[0]*cols for _ in range(rows)]
    right = [[0]*cols for _ in range(rows)]
    top = [[0]*cols for _ in range(rows)]
    bottom = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        row_vals = grid[r]
        lr = ring_1d(row_vals, fg)
        rr = ring_1d(row_vals[::-1], fg)[::-1]
        left[r] = lr; right[r] = rr
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        tr = ring_1d(col_vals, fg)
        br = ring_1d(col_vals[::-1], fg)[::-1]
        for r in range(rows):
            top[r][c] = tr[r]; bottom[r][c] = br[r]
    ring = [[min(left[r][c], right[r][c], top[r][c], bottom[r][c])
             for c in range(cols)] for r in range(rows)]
    return ring


def transition_depth(grid, bg_val, fg_val, rows, cols):
    INF = 999999
    depth = [[INF]*cols for _ in range(rows)]
    pq = []
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows-1 or c == 0 or c == cols-1:
                d = 0 if grid[r][c] == bg_val else 1
                if d < depth[r][c]:
                    depth[r][c] = d; heapq.heappush(pq, (d, r, c))
    while pq:
        d, r, c = heapq.heappop(pq)
        if d > depth[r][c]: continue
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nd = d + (1 if grid[nr][nc] != grid[r][c] else 0)
                if nd < depth[nr][nc]:
                    depth[nr][nc] = nd; heapq.heappush(pq, (nd, nr, nc))
    return depth


def collapse(seq):
    if not seq: return ()
    result = [seq[0]]
    for v in seq[1:]:
        if v != result[-1]: result.append(v)
    return tuple(result)


def get_interior(grid, bg_val, rows, cols):
    exterior = [[False]*cols for _ in range(rows)]
    q = deque()
    for r in range(rows):
        for c in [0, cols-1]:
            if grid[r][c] == bg_val and not exterior[r][c]:
                exterior[r][c] = True; q.append((r, c))
    for c in range(cols):
        for r in [0, rows-1]:
            if grid[r][c] == bg_val and not exterior[r][c]:
                exterior[r][c] = True; q.append((r, c))
    while q:
        cr, cc = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < rows and 0 <= nc < cols and not exterior[nr][nc] and grid[nr][nc] == bg_val:
                exterior[nr][nc] = True; q.append((nr, nc))
    return [[not exterior[r][c] for c in range(cols)] for r in range(rows)]


def dedup_merge_with_gaps(rprofs):
    unique = []; prev_had = False
    for rp in rprofs:
        if rp is None:
            if prev_had and unique: unique.append(None); prev_had = False
            continue
        prev_had = True
        if not unique or unique[-1] != rp: unique.append(rp)
    while unique and unique[-1] is None: unique.pop()
    merged = []
    for rp in unique:
        if rp is None: merged.append(None); continue
        mx = max(rp)
        if merged and merged[-1] is not None and max(merged[-1]) == mx:
            if len(rp) > len(merged[-1]): merged[-1] = rp
        else: merged.append(rp)
    if any(rp is None for rp in merged): merged.append(None)
    maxes = [max(rp) if rp else 0 for rp in merged]
    return merged, maxes


def compute_smooth_ring(maxes, peak_pos):
    """Compute smooth ring levels, skipping secondary bumps."""
    N = len(maxes)
    if N == 0: return []
    left_ring = [0] * N
    cur_ring = 0
    for i in range(peak_pos + 1):
        if maxes[i] > cur_ring:
            if all(maxes[j] >= maxes[i] for j in range(i, peak_pos + 1)):
                cur_ring = maxes[i]
        left_ring[i] = min(maxes[i], cur_ring)
    for i in range(peak_pos + 1, N):
        left_ring[i] = min(maxes[i], left_ring[i-1])
    right_ring = [0] * N
    cur_ring = 0
    for i in range(N - 1, peak_pos - 1, -1):
        if maxes[i] > cur_ring:
            if all(maxes[j] >= maxes[i] for j in range(peak_pos, i + 1)):
                cur_ring = maxes[i]
        right_ring[i] = min(maxes[i], cur_ring)
    for i in range(peak_pos - 1, -1, -1):
        right_ring[i] = min(maxes[i], right_ring[i+1])
    return [min(left_ring[i], right_ring[i]) for i in range(N)]


def solve(grid):
    rows, cols = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    cnt = Counter(flat)
    bg_val = cnt.most_common(1)[0][0]
    fg_val = [c for c in cnt if c != bg_val][0]

    ring = compute_ring_grid(grid, fg_val, rows, cols)
    td = transition_depth(grid, bg_val, fg_val, rows, cols)
    depth = [[min(ring[r][c], td[r][c]) for c in range(cols)] for r in range(rows)]
    interior = get_interior(grid, bg_val, rows, cols)

    # Row ring profiles from clamped depth
    row_rprofs = []
    for r in range(rows):
        ic = [c for c in range(cols) if interior[r][c]]
        if not ic: row_rprofs.append(None); continue
        rings = [depth[r][c] for c in range(ic[0], ic[-1]+1)]
        row_rprofs.append(collapse(rings))
    row_merged, row_mx = dedup_merge_with_gaps(row_rprofs)
    H = len(row_merged)

    # Column ring profiles
    col_rprofs = []
    for c in range(cols):
        ir = [r for r in range(rows) if interior[r][c]]
        if not ir: col_rprofs.append(None); continue
        rings = [depth[r][c] for r in range(ir[0], ir[-1]+1)]
        col_rprofs.append(collapse(rings))
    col_merged, col_mx = dedup_merge_with_gaps(col_rprofs)
    W = len(col_merged)

    # Gap boundaries (for extensions)
    row_gap = H
    for r in range(H):
        if row_merged[r] is None: row_gap = r; break
    col_gap = W
    for c in range(W):
        if col_merged[c] is None: col_gap = c; break

    row_mx_main = row_mx[:row_gap]
    col_mx_main = col_mx[:col_gap]

    peak_row = row_mx_main.index(max(row_mx_main)) if row_mx_main else 0
    peak_col = col_mx_main.index(max(col_mx_main)) if col_mx_main else 0

    smooth_col = compute_smooth_ring(col_mx_main, peak_col)
    smooth_row = compute_smooth_ring(row_mx_main, peak_row)

    output = []
    for r in range(H):
        if row_merged[r] is None:
            output.append([bg_val] * W); continue
        if r > row_gap:
            row = [bg_val] * W
            if r == row_gap + 1 and peak_col < W:
                row[peak_col] = fg_val
            output.append(row); continue

        rp = list(row_merged[r])
        is_peak = (r == peak_row)
        row = [bg_val] * W

        if is_peak:
            # Peak row: 1:1 mapping from ring profile
            for c in range(min(col_gap, len(rp))):
                row[c] = fg_val if rp[c] % 2 == 1 else bg_val
            # Extension columns from peak row's profile
            ext_rp = []; saw_zero = False
            for v in rp:
                if v == 0: saw_zero = True
                elif saw_zero: ext_rp.append(v)
            for c in range(col_gap, W):
                if col_merged[c] is not None and ext_rp:
                    ext_ci = sum(1 for ci in range(col_gap, c) if col_merged[ci] is None)
                    ext_idx = c - col_gap - ext_ci
                    if 0 <= ext_idx < len(ext_rp) and ext_rp[ext_idx] % 2 == 1:
                        row[c] = fg_val
        else:
            # Non-peak rows: ring = min(smooth_row, smooth_col)
            # At peak column, use actual row_max for secondary arm correctness
            sr = smooth_row[r] if r < len(smooth_row) else 0
            rm = row_mx_main[r] if r < len(row_mx_main) else 0
            for c in range(col_gap):
                sc = smooth_col[c] if c < len(smooth_col) else 0
                effective_row = rm if c == peak_col else sr
                ring_val = min(effective_row, sc)
                row[c] = fg_val if ring_val > 0 and ring_val % 2 == 1 else bg_val

        output.append(row)

    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/2d0172a1.json") as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Train {i}: PASS ✓")
        else:
            mismatches = sum(1 for r in range(len(expected)) for c in range(len(expected[0])) if result[r][c] != expected[r][c])
            print(f"Train {i}: FAIL - {mismatches} mismatches")
            all_pass = False

    for i, ex in enumerate(data["test"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Test {i}: PASS ✓")
        else:
            mismatches = sum(1 for r in range(len(expected)) for c in range(len(expected[0])) if result[r][c] != expected[r][c])
            print(f"Test {i}: FAIL - {mismatches} mismatches")
            all_pass = False

    if all_pass:
        print("\nAll examples pass! ✓")
