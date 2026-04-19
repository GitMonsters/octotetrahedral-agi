from collections import Counter, deque
import copy

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = copy.deepcopy(grid)
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    non_bg = [v for v in flat if v != bg]
    fill = Counter(non_bg).most_common(1)[0][0] if non_bg else bg

    # Find connected non-bg components
    visited = [[False]*cols for _ in range(rows)]
    components = []
    def bfs(sr, sc):
        q = deque([(sr, sc)]); visited[sr][sc] = True; comp = [(sr, sc)]
        while q:
            r, c = q.popleft()
            for nr, nc in [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]:
                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] != bg:
                    visited[nr][nc] = True; q.append((nr, nc)); comp.append((nr, nc))
        return comp
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                components.append(bfs(r, c))

    rects = []; marker_comp = None
    for comp in components:
        rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
        rmin, rmax, cmin, cmax = min(rs), max(rs), min(cs), max(cs)
        bbox = (rmax-rmin+1)*(cmax-cmin+1)
        fc = sum(1 for r,c in comp if grid[r][c] == fill)
        if fc > bbox * 0.5 and len(comp) >= 20:
            rects.append((rmin, rmax, cmin, cmax, comp))
        elif marker_comp is None or len(comp) < len(marker_comp):
            marker_comp = comp
    if not marker_comp: return out

    # Find marker cells (non-bg, non-fill)
    marker_special = [(r, c, grid[r][c]) for r, c in marker_comp if grid[r][c] != fill]
    marker_fill_cells = [(r, c) for r, c in marker_comp if grid[r][c] == fill]

    # Identify anchor_color: appears in marker AND inside rects
    other_colors = set(v for _,_,v in marker_special) - {bg}
    anchor_color = None
    for color in other_colors:
        for rmin, rmax, cmin, cmax, comp in rects:
            if any(grid[r][c] == color for r, c in comp):
                anchor_color = color; break
        if anchor_color: break

    # Anchor in marker
    anchor_r = anchor_c = None
    if anchor_color:
        for r, c, v in marker_special:
            if v == anchor_color: anchor_r, anchor_c = r, c; break
    if anchor_r is None and marker_fill_cells:
        anchor_r, anchor_c = marker_fill_cells[0]
    if anchor_r is None:
        mrs = [r for r,c in marker_comp]; mcs = [c for r,c in marker_comp]
        anchor_r, anchor_c = (min(mrs)+max(mrs))//2, (min(mcs)+max(mcs))//2

    # Pattern cells relative to anchor (non-bg, non-fill, non-anchor)
    pat_cells = []
    for r, c, v in marker_special:
        if v != anchor_color:
            pat_cells.append((r - anchor_r, c - anchor_c, v))

    # Detect line directions
    h_line = set(); v_line = set()
    for dr, dc, v in pat_cells:
        if dr == 0 and dc < 0: h_line.add('left')
        if dr == 0 and dc > 0: h_line.add('right')
        if dc == 0 and dr < 0: v_line.add('up')
        if dc == 0 and dr > 0: v_line.add('down')

    is_h_line = len(h_line) == 1  # only left or only right
    is_v_line = len(v_line) == 1

    # bg trail in marker (bg cells adjacent to anchor within marker bbox)
    mrs = [r for r,c in marker_comp]; mcs = [c for r,c in marker_comp]
    m_rmin, m_rmax, m_cmin, m_cmax = min(mrs), max(mrs), min(mcs), max(mcs)
    bg_trail_dir = None  # direction from anchor where bg is
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        nr, nc = anchor_r + dr, anchor_c + dc
        if m_rmin <= nr <= m_rmax and m_cmin <= nc <= m_cmax:
            if grid[nr][nc] == bg and (nr, nc) not in set((r,c) for r,c in marker_comp):
                # bg cell in marker bbox but not in component
                if dc == 0 and dr > 0: bg_trail_dir = 'down'
                elif dc == 0 and dr < 0: bg_trail_dir = 'up'
                elif dr == 0 and dc > 0: bg_trail_dir = 'right'
                elif dr == 0 and dc < 0: bg_trail_dir = 'left'

    # Erase marker
    for r in range(m_rmin, m_rmax+1):
        for c in range(m_cmin, m_cmax+1):
            out[r][c] = bg

    # Process each rect
    for rmin, rmax, cmin, cmax, comp in rects:
        ar = ac = None
        if anchor_color:
            for r, c in comp:
                if grid[r][c] == anchor_color:
                    ar, ac = r, c; break
        if ar is None:
            # Reflect
            if anchor_r < rmin: ar = 2*rmin - anchor_r
            elif anchor_r > rmax: ar = 2*rmax - anchor_r
            else: ar = anchor_r
            if anchor_c < cmin: ac = 2*cmin - anchor_c
            elif anchor_c > cmax: ac = 2*cmax - anchor_c
            else: ac = anchor_c
            ar = max(rmin+1, min(rmax-1, ar))
            ac = max(cmin+1, min(cmax-1, ac))

        # Paint pattern cells at anchor
        for dr, dc, v in pat_cells:
            nr, nc = ar + dr, ac + dc
            if rmin <= nr <= rmax and cmin <= nc <= cmax:
                out[nr][nc] = v

        # Extend lines
        if is_h_line and pat_cells:
            pc = pat_cells[0][2]
            d = list(h_line)[0]
            if d == 'left':
                for c in range(ac - 1, cmin - 1, -1): out[ar][c] = pc
            else:
                for c in range(ac + 1, cmax + 1): out[ar][c] = pc

        if is_v_line and pat_cells:
            pc = pat_cells[0][2]
            d = list(v_line)[0]
            if d == 'up':
                for r in range(ar - 1, rmin - 1, -1): out[r][ac] = pc
            else:
                for r in range(ar + 1, rmax + 1): out[r][ac] = pc

        # If line extends, add bg bracket on opposite side
        if (is_h_line or is_v_line) and not (is_h_line and is_v_line):
            # Single line direction - add bg bracket
            if is_h_line:
                d = list(h_line)[0]
                opp = 1 if d == 'left' else -1
                for dr in [-1, 0, 1]:
                    for k in range(1, max(rmax-rmin, cmax-cmin)+1):
                        nc = ac + opp * k
                        if cmin <= nc <= cmax:
                            nr = ar + dr
                            if rmin <= nr <= rmax:
                                out[nr][nc] = bg
                        else:
                            break
            elif is_v_line:
                d = list(v_line)[0]
                opp = 1 if d == 'up' else -1
                for dc in [-1, 0, 1]:
                    for k in range(1, max(rmax-rmin, cmax-cmin)+1):
                        nr = ar + opp * k
                        if rmin <= nr <= rmax:
                            nc = ac + dc
                            if cmin <= nc <= cmax:
                                out[nr][nc] = bg
                        else:
                            break

        # bg trail
        if bg_trail_dir:
            if bg_trail_dir == 'down':
                for r in range(ar+1, rmax+1): out[r][ac] = bg
            elif bg_trail_dir == 'up':
                for r in range(ar-1, rmin-1, -1): out[r][ac] = bg
            elif bg_trail_dir == 'right':
                for c in range(ac+1, cmax+1): out[ar][c] = bg
            elif bg_trail_dir == 'left':
                for c in range(ac-1, cmin-1, -1): out[ar][c] = bg

        # Restore anchor
        if anchor_color: out[ar][ac] = anchor_color

    return out
