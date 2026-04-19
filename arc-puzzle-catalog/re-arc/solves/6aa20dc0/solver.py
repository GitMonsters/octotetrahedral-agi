def transform(input_grid):
    """
    Rule: There's a small template pattern with 3 non-bg colors: a connector color
    and 2 marker colors (each appearing once). Outside the template are pairs of
    monochromatic blocks matching the marker colors. For each pair, the template is
    scaled to block size, transformed (rotation/flip) so markers align with blocks,
    and stamped onto the grid.
    """
    from collections import Counter, deque
    from itertools import permutations

    grid = [row[:] for row in input_grid]
    H, W = len(grid), len(grid[0])

    # Background = most common color
    flat = [grid[r][c] for r in range(H) for c in range(W)]
    bg = Counter(flat).most_common(1)[0][0]

    # Non-bg cells
    non_bg = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                non_bg[(r, c)] = grid[r][c]

    # 8-connected components of non-bg cells
    visited = set()
    components = []
    for pos in non_bg:
        if pos in visited:
            continue
        q = deque([pos])
        visited.add(pos)
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nb = (r + dr, c + dc)
                    if nb in non_bg and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
        components.append(comp)

    # Template = component with 3+ distinct colors
    template_comp = None
    for comp in components:
        colors = set(non_bg[p] for p in comp)
        if len(colors) >= 3:
            template_comp = comp
            break

    # Extract template grid from bounding box
    t_min_r = min(r for r, c in template_comp)
    t_max_r = max(r for r, c in template_comp)
    t_min_c = min(c for r, c in template_comp)
    t_max_c = max(c for r, c in template_comp)
    t_h = t_max_r - t_min_r + 1
    t_w = t_max_c - t_min_c + 1

    tmpl = [[bg] * t_w for _ in range(t_h)]
    for r, c in template_comp:
        tmpl[r - t_min_r][c - t_min_c] = grid[r][c]

    # Connector = most common non-bg color in template; markers = colors appearing once
    tcounts = Counter()
    for r in range(t_h):
        for c in range(t_w):
            if tmpl[r][c] != bg:
                tcounts[tmpl[r][c]] += 1
    connector = max(tcounts, key=tcounts.get)
    markers = sorted(col for col, cnt in tcounts.items() if cnt == 1)
    marker_a, marker_b = markers[0], markers[1]

    # All 8 orientations of template (4 rotations x 2 flips)
    def get_orientations(t):
        res = []
        cur = [row[:] for row in t]
        for _ in range(4):
            res.append([row[:] for row in cur])
            res.append([row[::-1] for row in cur])
            cur = [list(r) for r in zip(*cur[::-1])]  # 90° CW
        return res

    all_orients = get_orientations(tmpl)

    # Find marker objects outside template
    template_set = set(template_comp)
    marker_objs = []
    for comp in components:
        if set(comp) & template_set:
            continue
        col = non_bg[comp[0]]
        if col not in markers:
            continue
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        marker_objs.append({
            'color': col,
            'top': min(rs),
            'left': min(cs),
            'h': max(rs) - min(rs) + 1,
            'w': max(cs) - min(cs) + 1,
        })

    # Group by block size
    by_size = {}
    for obj in marker_objs:
        key = (obj['h'], obj['w'])
        by_size.setdefault(key, []).append(obj)

    output = [row[:] for row in input_grid]

    for (bh, bw), objs in by_size.items():
        a_list = [o for o in objs if o['color'] == marker_a]
        b_list = [o for o in objs if o['color'] == marker_b]
        if not a_list or not b_list:
            continue

        # Try all pairings of a with b
        best_stamps = None
        for perm in permutations(range(len(b_list))):
            stamps = []
            ok = True
            for i, a_obj in enumerate(a_list):
                if i >= len(perm):
                    ok = False
                    break
                b_obj = b_list[perm[i]]
                found = False
                for orient in all_orients:
                    oh, ow = len(orient), len(orient[0])
                    atp = btp = None
                    for r in range(oh):
                        for c in range(ow):
                            if orient[r][c] == marker_a and atp is None:
                                atp = (r, c)
                            if orient[r][c] == marker_b and btp is None:
                                btp = (r, c)
                    if atp is None or btp is None:
                        continue
                    r0a = a_obj['top'] - atp[0] * bh
                    c0a = a_obj['left'] - atp[1] * bw
                    r0b = b_obj['top'] - btp[0] * bh
                    c0b = b_obj['left'] - btp[1] * bw
                    if r0a == r0b and c0a == c0b:
                        stamps.append((orient, r0a, c0a))
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok and len(stamps) == len(a_list):
                best_stamps = stamps
                break

        if best_stamps:
            for orient, r0, c0 in best_stamps:
                oh, ow = len(orient), len(orient[0])
                for tr in range(oh):
                    for tc in range(ow):
                        color = orient[tr][tc]
                        if color == bg:
                            continue
                        for dr in range(bh):
                            for dc in range(bw):
                                gr = r0 + tr * bh + dr
                                gc = c0 + tc * bw + dc
                                if 0 <= gr < H and 0 <= gc < W:
                                    output[gr][gc] = color

    return output


# ── Testing ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    examples = [
        {
            "input": [[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,1,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,4,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,2,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,2,2,2,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,2,2,2,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,2,2,2,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,6,6,6,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,6,6,6,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,6,6,6,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]],
            "output": [[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,1,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,4,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,2,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,1,1,1,1,1,1,2,2,2,4,4,4,4,4,4,4],[4,4,4,4,4,1,1,1,1,1,1,2,2,2,4,4,4,4,4,4,4],[4,4,4,4,4,1,1,1,1,1,1,2,2,2,4,4,4,4,4,4,4],[4,4,4,4,4,1,1,1,4,4,4,1,1,1,4,4,4,4,4,4,4],[4,4,4,4,4,1,1,1,4,4,4,1,1,1,4,4,4,4,4,4,4],[4,4,4,4,4,1,1,1,4,4,4,1,1,1,4,4,4,4,4,4,4],[4,4,4,4,4,6,6,6,1,1,1,1,1,1,4,4,4,4,4,4,4],[4,4,4,4,4,6,6,6,1,1,1,1,1,1,4,4,4,4,4,4,4],[4,4,4,4,4,6,6,6,1,1,1,1,1,1,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]]
        },
        {
            "input": [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,1,1],[1,1,1,1,1,2,8,8,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,8,8,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,8,1,3,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],
            "output": [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,2,8,8,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,8,1,3,1,1,1],[1,1,1,1,1,2,8,8,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,8,8,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,8,1,3,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,8,8,8,8,2,2,1,1,1,1,1,1,1,1],[1,1,1,1,1,8,8,8,8,2,2,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,8,8,8,8,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,8,8,8,8,1,1,1,1,1,1,1,1],[1,1,1,1,1,3,3,1,1,8,8,1,1,1,1,1,1,1,1],[1,1,1,1,1,3,3,1,1,8,8,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
        },
        {
            "input": [[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,2,3,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,3,3,3,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,3,4,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,4,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,2,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
            "output": [[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,2,3,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,3,3,3,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,3,4,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,4,3,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,3,3,3,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,3,2,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,3,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,3,3,3,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,4,3,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]]
        }
    ]

    all_pass = True
    for i, ex in enumerate(examples):
        result = transform(ex["input"])
        if result == ex["output"]:
            print(f"Example {i}: PASS")
        else:
            print(f"Example {i}: FAIL")
            all_pass = False
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != ex["output"][r][c]:
                        print(f"  Diff at ({r},{c}): got {result[r][c]}, expected {ex['output'][r][c]}")

    if all_pass:
        print("SOLVED")
    else:
        print("FAILED")
