import json
import os
from collections import Counter


def _solve_kernels(task_data):
    """Solve for K_HIGH and K_LOW kernels from training data using GF(2) linear algebra."""
    dr_min, dr_max = -30, 30
    dc_min, dc_max = -25, 25
    dc_range = dc_max - dc_min + 1

    def var_idx(dr, dc):
        return (dr - dr_min) * dc_range + (dc - dc_min)

    # Determine hierarchy for each training example
    hierarchies = {}
    for idx, ex in enumerate(task_data['train']):
        inp = ex['input']
        out = ex['output']
        R, C = len(inp), len(inp[0])

        counts = Counter(v for row in inp for v in row)
        colors_sorted = counts.most_common()
        marker_color = colors_sorted[-1][0]
        non_marker = [c for c, _ in colors_sorted if c != marker_color]

        # Determine which non-marker color is "high" vs "low"
        # Low = most common neighbor of markers
        markers = [(r, c) for r in range(R) for c in range(C) if inp[r][c] == marker_color]
        neighbor_counts = Counter()
        for mr, mc in markers:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = mr + dr, mc + dc
                if 0 <= nr < R and 0 <= nc < C and inp[nr][nc] != marker_color:
                    neighbor_counts[inp[nr][nc]] += 1

        if neighbor_counts:
            low_color = neighbor_counts.most_common(1)[0][0]
        else:
            low_color = non_marker[-1]
        high_color = [c for c in non_marker if c != low_color][0]

        hierarchies[idx] = (high_color, low_color, marker_color)

    kernels = {}
    for color_type in ["high", "low"]:
        equations = []
        for idx, ex in enumerate(task_data['train']):
            inp = ex['input']
            out = ex['output']
            R, C = len(inp), len(inp[0])
            high, low, mc = hierarchies[idx]

            markers = [(r, c) for r in range(R) for c in range(C) if inp[r][c] == mc]
            target_color = high if color_type == "high" else low

            for r in range(R):
                for c in range(C):
                    if inp[r][c] != target_color:
                        continue
                    target = 1 if inp[r][c] != out[r][c] else 0
                    var_ids = []
                    for mr, mcol in markers:
                        vi = var_idx(r - mr, c - mcol)
                        var_ids.append(vi)
                    equations.append((var_ids, target))

        # Map variables to compact indices
        used_vars = set()
        for var_ids, _ in equations:
            used_vars.update(var_ids)
        used_vars_list = sorted(used_vars)
        vmap = {v: i for i, v in enumerate(used_vars_list)}
        num_used = len(used_vars_list)
        n_eq = len(equations)

        # Build augmented matrix as list of lists (GF(2))
        # Use bitsets for efficiency
        aug = []
        for var_ids, target in equations:
            row_bits = [0] * (num_used + 1)
            for v in var_ids:
                row_bits[vmap[v]] ^= 1
            row_bits[num_used] = target
            aug.append(row_bits)

        # Gaussian elimination over GF(2)
        pivot_cols = []
        cur_row = 0
        for col in range(num_used):
            # Find pivot
            pivot_r = None
            for r in range(cur_row, n_eq):
                if aug[r][col] == 1:
                    pivot_r = r
                    break
            if pivot_r is None:
                continue
            aug[cur_row], aug[pivot_r] = aug[pivot_r], aug[cur_row]
            for r in range(n_eq):
                if r != cur_row and aug[r][col] == 1:
                    for j in range(num_used + 1):
                        aug[r][j] ^= aug[cur_row][j]
            pivot_cols.append(col)
            cur_row += 1

        # Extract solution
        x = [0] * num_used
        for i in range(len(pivot_cols) - 1, -1, -1):
            col = pivot_cols[i]
            x[col] = aug[i][num_used]

        K = {}
        for i, vi in enumerate(used_vars_list):
            dr = vi // dc_range + dr_min
            dc = vi % dc_range + dc_min
            if x[i] == 1:
                K[(dr, dc)] = 1

        kernels[color_type] = K

    return kernels["high"], kernels["low"], hierarchies


def _load_task():
    task_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '032a299a.json')
    with open(task_path) as f:
        return json.load(f)


_task_data = _load_task()
_K_HIGH, _K_LOW, _hierarchies = _solve_kernels(_task_data)


def transform(input_grid):
    R = len(input_grid)
    C = len(input_grid[0]) if R > 0 else 0

    counts = Counter(input_grid[r][c] for r in range(R) for c in range(C))
    marker_color = min(counts, key=counts.get)

    markers = [(r, c) for r in range(R) for c in range(C) if input_grid[r][c] == marker_color]

    non_marker = [c for c, _ in counts.most_common() if c != marker_color]

    # Determine high/low by checking marker neighbors
    neighbor_counts = Counter()
    for mr, mc in markers:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = mr + dr, mc + dc
            if 0 <= nr < R and 0 <= nc < C and input_grid[nr][nc] != marker_color:
                neighbor_counts[input_grid[nr][nc]] += 1

    if neighbor_counts:
        low_color = neighbor_counts.most_common(1)[0][0]
    else:
        low_color = non_marker[-1] if non_marker else 0
    high_color = [c for c in non_marker if c != low_color][0] if len(non_marker) > 1 else non_marker[0]

    out = [row[:] for row in input_grid]

    for r in range(R):
        for c in range(C):
            if input_grid[r][c] == marker_color:
                continue

            K = _K_HIGH if input_grid[r][c] == high_color else _K_LOW

            xor_val = 0
            for mr, mc in markers:
                dr = r - mr
                dc = c - mc
                if (dr, dc) in K:
                    xor_val ^= 1

            if xor_val == 1:
                if input_grid[r][c] == high_color:
                    out[r][c] = low_color
                else:
                    out[r][c] = marker_color

    return out
