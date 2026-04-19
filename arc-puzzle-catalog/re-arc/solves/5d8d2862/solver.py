def transform(input_grid):
    import copy
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])
    output = copy.deepcopy(input_grid)

    flat = [c for r in input_grid for c in r]
    bg = Counter(flat).most_common(1)[0][0]

    # Find connected components (8-connected)
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg and (r, c) not in visited:
                comp = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr, cc, input_grid[cr][cc]))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and input_grid[nr][nc] != bg:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                components.append(comp)

    # Key component = most unique colors
    key_idx = max(range(len(components)),
                  key=lambda i: len(set(v for _, _, v in components[i])))
    key_comp = components[key_idx]
    key_pos_color = {(r, c): v for r, c, v in key_comp}

    # 8 orientations of the dihedral group
    def apply_orient(oi, r, c):
        if oi == 0: return (r, c)
        if oi == 1: return (-c, r)
        if oi == 2: return (-r, -c)
        if oi == 3: return (c, -r)
        if oi == 4: return (r, -c)
        if oi == 5: return (-r, c)
        if oi == 6: return (c, r)
        if oi == 7: return (-c, -r)

    inv_map = [0, 3, 2, 1, 4, 5, 6, 7]

    for i, comp in enumerate(components):
        if i == key_idx:
            continue

        found = False
        for oi in range(8):
            if found:
                break

            oriented = [(apply_orient(oi, r, c), v) for r, c, v in comp]

            for (or_, oc_), ov in oriented:
                if found:
                    break
                for kr, kc, kv in key_comp:
                    if kv != ov:
                        continue

                    dr = kr - or_
                    dc = kc - oc_

                    match = True
                    for (opr, opc), opv in oriented:
                        tr, tc = opr + dr, opc + dc
                        if key_pos_color.get((tr, tc)) != opv:
                            match = False
                            break

                    if match:
                        matched = set()
                        for (opr, opc), _ in oriented:
                            matched.add((opr + dr, opc + dc))

                        decoration = [(r, c, v) for r, c, v in key_comp
                                      if (r, c) not in matched]

                        inv_oi = inv_map[oi]
                        for dkr, dkc, dv in decoration:
                            ur, uc = dkr - dr, dkc - dc
                            orig_r, orig_c = apply_orient(inv_oi, ur, uc)
                            if 0 <= orig_r < rows and 0 <= orig_c < cols:
                                output[orig_r][orig_c] = dv

                        found = True
                        break

    return output
