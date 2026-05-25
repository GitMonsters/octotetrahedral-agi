def solve(grid: list[list[int]]) -> list[list[int]]:
    """Connect rectangles via minimum spanning tree of gap distances.

    Rectangles outlined with 3s are found, then pairs that face each other
    along an axis (with overlapping inner ranges and a clear gap) form candidate
    edges. A MST by gap distance selects which pairs to connect. Each selected
    pair gets a tunnel: bridge walls (3s) on the first/last overlapping inner
    rows/cols, with facing walls opened (3→0) on middle rows/cols.
    """
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    rects = _find_rectangles(grid, rows, cols)
    edges = _find_edges(grid, rects)
    mst_edges = _kruskal(len(rects), edges)

    for i, j, direction in mst_edges:
        if direction == 'h':
            _build_horizontal(result, grid, rects[i], rects[j])
        else:
            _build_vertical(result, grid, rects[i], rects[j])

    return result


class _UF:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        self.parent[a] = b
        return True


def _find_edges(grid, rects):
    """Find all valid candidate edges between rectangle pairs."""
    edges = []
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            d = _h_gap(grid, rects[i], rects[j])
            if d is not None:
                edges.append((d, i, j, 'h'))
            d = _v_gap(grid, rects[i], rects[j])
            if d is not None:
                edges.append((d, i, j, 'v'))
    edges.sort()
    return edges


def _kruskal(n, edges):
    """Return MST edges as (i, j, direction) triples."""
    uf = _UF(n)
    mst = []
    for _, i, j, d in edges:
        if uf.union(i, j):
            mst.append((i, j, d))
    return mst


def _h_gap(grid, a, b):
    """Return horizontal gap distance, or None if not connectable."""
    at, ab_, al, ar = a
    bt, bb, bl, br = b
    if ar < bl:
        left, right = a, b
    elif br < al:
        left, right = b, a
    else:
        return None
    lt, lb, ll, lr = left
    rt, rb, rl, rr = right
    o_top = max(lt + 1, rt + 1)
    o_bot = min(lb - 1, rb - 1)
    if o_top > o_bot:
        return None
    gl, gr = lr + 1, rl - 1
    if gl > gr:
        return None
    for r in range(o_top, o_bot + 1):
        for c in range(gl, gr + 1):
            if grid[r][c] != 0:
                return None
    return gr - gl + 1


def _v_gap(grid, a, b):
    """Return vertical gap distance, or None if not connectable."""
    at, ab_, al, ar = a
    bt, bb, bl, br = b
    if ab_ < bt:
        top, bot = a, b
    elif bb < at:
        top, bot = b, a
    else:
        return None
    tt, tb, tl, tr = top
    bt2, bb2, bl2, br2 = bot
    o_left = max(tl + 1, bl2 + 1)
    o_right = min(tr - 1, br2 - 1)
    if o_left > o_right:
        return None
    gt, gb = tb + 1, bt2 - 1
    if gt > gb:
        return None
    for r in range(gt, gb + 1):
        for c in range(o_left, o_right + 1):
            if grid[r][c] != 0:
                return None
    return gb - gt + 1


def _find_rectangles(grid, rows, cols):
    visited = [[False] * cols for _ in range(rows)]
    rects = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3 and not visited[r][c]:
                queue = [(r, c)]
                visited[r][c] = True
                component = []
                while queue:
                    cr, cc = queue.pop(0)
                    component.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 3:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                min_r = min(p[0] for p in component)
                max_r = max(p[0] for p in component)
                min_c = min(p[1] for p in component)
                max_c = max(p[1] for p in component)
                rects.append((min_r, max_r, min_c, max_c))
    return rects


def _build_horizontal(result, grid, a, b):
    """Build a horizontal tunnel between two rectangles."""
    a_top, a_bot, a_left, a_right = a
    b_top, b_bot, b_left, b_right = b

    if a_right < b_left:
        left, right = a, b
    elif b_right < a_left:
        left, right = b, a
    else:
        return

    l_top, l_bot, l_left, l_right = left
    r_top, r_bot, r_left, r_right = right

    overlap_top = max(l_top + 1, r_top + 1)
    overlap_bot = min(l_bot - 1, r_bot - 1)
    gap_left = l_right + 1
    gap_right = r_left - 1

    for c in range(gap_left, gap_right + 1):
        result[overlap_top][c] = 3
        result[overlap_bot][c] = 3

    for r in range(overlap_top + 1, overlap_bot):
        result[r][l_right] = 0
        result[r][r_left] = 0


def _build_vertical(result, grid, a, b):
    """Build a vertical tunnel between two rectangles."""
    a_top, a_bot, a_left, a_right = a
    b_top, b_bot, b_left, b_right = b

    if a_bot < b_top:
        top, bot = a, b
    elif b_bot < a_top:
        top, bot = b, a
    else:
        return

    t_top, t_bot, t_left, t_right = top
    b_top2, b_bot2, b_left2, b_right2 = bot

    overlap_left = max(t_left + 1, b_left2 + 1)
    overlap_right = min(t_right - 1, b_right2 - 1)
    gap_top = t_bot + 1
    gap_bot = b_top2 - 1

    for r in range(gap_top, gap_bot + 1):
        result[r][overlap_left] = 3
        result[r][overlap_right] = 3

    for c in range(overlap_left + 1, overlap_right):
        result[t_bot][c] = 0
        result[b_top2][c] = 0
