"""
ARC Task 212709ed Solver

Two cases:
1. Single non-bg color: an L-tromino (TL+TR+BR, missing BL) is extended into a
   fractal-like pattern with scaled copies and scattered K=1 companions.
2. Two non-bg colors: a template (shape_color cell + adjacent ext_color cells)
   defines an extension pattern applied to all shape_color rectangles, scaled by
   their size.
"""

from collections import Counter, deque
import numpy as np
from scipy import ndimage


def transform(grid):
    grid = [list(row) for row in grid]
    H, W = len(grid), len(grid[0])
    counts = Counter(v for row in grid for v in row)
    bg = counts.most_common(1)[0][0]
    non_bg_colors = [c for c in counts if c != bg]

    if len(non_bg_colors) == 1:
        return _single_color(grid, H, W, bg, non_bg_colors[0])
    else:
        return _multi_color(grid, H, W, bg)


def _place_K_L(out, H, W, r_anchor, c_anchor, K, fg):
    """Place K-scale L-tromino (TL+TR+BR, missing BL) at anchor (r_anchor, c_anchor)."""
    for dr in range(K):
        for dc in range(K):
            for rr, cc in [
                (r_anchor + dr, c_anchor + dc),        # TL block
                (r_anchor + dr, c_anchor + K + dc),    # TR block
                (r_anchor + K + dr, c_anchor + K + dc) # BR block
            ]:
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr][cc] = fg


def _single_color(grid, H, W, bg, fg):
    """Handle single non-bg color: fractal L-tromino expansion."""
    out = [row[:] for row in grid]

    # Find the L-tromino anchor (top-left corner)
    cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == fg]
    r_a = min(r for r, c in cells)
    c_a = min(c for r, c in cells if r == r_a)

    # K_up: largest power of 2 s.t. 2*K_up <= r_a (space above)
    K_up = 0
    k = 1
    while 2 * k <= r_a:
        K_up = k
        k *= 2
    if K_up == 0:
        return out

    # Direction: go RIGHT if space; else LEFT
    right_ok = (c_a + 4 * K_up - 1 < W)
    direction = 'right' if right_ok else 'left'

    if direction == 'right':
        # --- K_up L: upper-right ---
        r_up = r_a - 2 * K_up
        c_up = c_a + 2 * K_up - 1
        _place_K_L(out, H, W, r_up, c_up, K_up, fg)

        # --- K_right L: lower-right of original ---
        K_right = 2
        while K_right > 0 and r_a + 2 * K_right >= H:
            K_right //= 2
        r_kr = c_kr = -1
        if K_right > 0:
            r_kr = r_a + 1
            c_kr = c_a + (K_right + 1) * 2
            if c_kr + 2 * K_right <= W:
                _place_K_L(out, H, W, r_kr, c_kr, K_right, fg)
            else:
                K_right = 0

        # --- K_left L: lower-left corner ---
        K_left = 2
        while K_left > 0 and H - 1 - 2 * K_left < r_a + 2:
            K_left //= 2
        r_kl = c_kl = -1
        if K_left > 0:
            r_kl = H - 1 - 2 * K_left
            c_kl = max(0, c_a - 2 * K_left)
            _place_K_L(out, H, W, r_kl, c_kl, K_left, fg)

        # --- K=1 right-edge L ---
        if W - 2 >= 0 and W - 2 != c_a:
            _place_K_L(out, H, W, r_a, W - 2, 1, fg)

        # --- Companion K=1 Ls from K_right L ---
        if K_right > 0 and c_kr >= 0:
            _place_K_L(out, H, W, r_kr, c_kr + 2 * K_right, 1, fg)
            _place_K_L(out, H, W, r_kr + K_right * 2 - 1, c_kr + 3 * K_right, 1, fg)

        # --- Companion K=1 Ls from K_left L ---
        if K_left > 0:
            _place_K_L(out, H, W, r_kl, W - 1 - 2 * K_left, 1, fg)
            if K_right > 0 and c_kr >= 0:
                _place_K_L(out, H, W, r_kl - 1, c_kr + 2, 1, fg)

        # --- UL scattered K=1 Ls (triangular number positions) ---
        for i in range(K_up):
            dr = i * (i + 1) // 2  # triangular number
            if i == 0:
                dc = -2 * K_up
            elif i <= K_up // 2:
                dc = -(K_up + 1)
            else:
                dc = -(K_up - 1)
            r_ul = r_up + dr
            c_ul = c_up + dc
            if 0 <= r_ul < H and 0 <= c_ul < W:
                _place_K_L(out, H, W, r_ul, c_ul, 1, fg)

    else:  # direction == 'left'
        # --- K_up L: upper-left ---
        r_up = r_a - 2 * K_up
        c_up = c_a - 2 * K_up + 1
        _place_K_L(out, H, W, r_up, c_up, K_up, fg)

        # --- K_left L: lower-left of original ---
        K_left = 2
        while K_left > 0 and r_a + 2 * K_left >= H:
            K_left //= 2
        r_kl = c_kl = -1
        if K_left > 0:
            r_kl = r_a + 1
            c_kl = c_a - (K_left + 1) * 2 + 1
            if c_kl >= 0:
                _place_K_L(out, H, W, r_kl, c_kl, K_left, fg)
            else:
                K_left = 0

        # --- K_right L: lower-right corner ---
        K_right = 2
        while K_right > 0 and H - 1 - 2 * K_right < r_a + 2:
            K_right //= 2
        r_kr = c_kr = -1
        if K_right > 0:
            r_kr = H - 1 - 2 * K_right
            c_kr = min(W - 2 * K_right, c_a + 2 * K_right)
            _place_K_L(out, H, W, r_kr, c_kr, K_right, fg)

        # --- K=1 left-edge L ---
        if 0 != c_a:
            _place_K_L(out, H, W, r_a, 0, 1, fg)

        # --- Companion K=1 Ls from K_left L ---
        if K_left > 0 and c_kl >= 0:
            _place_K_L(out, H, W, r_kl, c_kl - 2 * K_left, 1, fg)
            _place_K_L(out, H, W, r_kl + K_left * 2 - 1, c_kl - 3 * K_left, 1, fg)

        # --- Companion K=1 Ls from K_right L ---
        if K_right > 0:
            _place_K_L(out, H, W, r_kr, 2 * K_right - 1, 1, fg)
            if K_left > 0 and c_kl >= 0:
                _place_K_L(out, H, W, r_kr - 1, c_kl - 2, 1, fg)

        # --- UR scattered K=1 Ls (triangular, mirrored) ---
        for i in range(K_up):
            dr = i * (i + 1) // 2
            if i == 0:
                dc = 2 * K_up - 1
            elif i <= K_up // 2:
                dc = K_up
            else:
                dc = K_up - 2
            r_ur = r_up + dr
            c_ur = c_up + 2 * K_up - 1 + dc
            if 0 <= r_ur < H and 0 <= c_ur < W:
                _place_K_L(out, H, W, r_ur, c_ur, 1, fg)

    return out


def _multi_color(grid, H, W, bg):
    """Handle two non-bg colors: template-based extension."""
    arr = np.array(grid)
    out = arr.copy()
    counts = Counter(v for row in grid for v in row)
    non_bg = [c for c in counts if c != bg]

    # Identify shape_color (more frequent) and ext_color (less frequent)
    shape_color = max(non_bg, key=lambda c: counts[c])
    ext_color = min(non_bg, key=lambda c: counts[c])

    # Find template anchor: shape_color cell adjacent to ext_color cell
    template_anchor = None
    for r in range(H):
        for c in range(W):
            if arr[r, c] == shape_color:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and arr[nr, nc] == ext_color:
                        template_anchor = (r, c)
                        break
            if template_anchor:
                break
        if template_anchor:
            break

    if template_anchor is None:
        return grid

    # BFS from template_anchor through ext_color neighbors to find all template ext cells
    ta_r, ta_c = template_anchor
    ext_cells = []
    visited = {(ta_r, ta_c)}
    queue = deque([(ta_r, ta_c)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and arr[nr, nc] == ext_color:
                visited.add((nr, nc))
                ext_cells.append((nr, nc))
                queue.append((nr, nc))

    template_offsets = [(er - ta_r, ec - ta_c) for er, ec in ext_cells]

    # Apply template to each non-template shape component, scaled by component size
    shape_mask = (arr == shape_color).astype(int)
    labeled, n_comps = ndimage.label(shape_mask)
    template_comp = int(labeled[ta_r, ta_c])

    for comp_id in range(1, n_comps + 1):
        if comp_id == template_comp:
            continue
        coords = np.argwhere(labeled == comp_id)
        r_min = int(coords[:, 0].min())
        c_min = int(coords[:, 1].min())
        r_max = int(coords[:, 0].max())
        c_max = int(coords[:, 1].max())
        K = max(r_max - r_min + 1, c_max - c_min + 1)

        for dr, dc in template_offsets:
            for delta_r in range(K):
                for delta_c in range(K):
                    nr = r_min + dr * K + delta_r
                    nc = c_min + dc * K + delta_c
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr][nc] = ext_color

    return out.tolist()
