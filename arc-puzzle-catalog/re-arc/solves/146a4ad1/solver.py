from collections import Counter


def transform(grid: list[list[int]]) -> list[list[int]]:
    inp = grid
    bg = Counter(v for row in inp for v in row).most_common(1)[0][0]
    in_c: dict = {}
    for i, row in enumerate(inp):
        for j, v in enumerate(row):
            if v != bg:
                in_c.setdefault(v, []).append((i, j))
    H, W = len(inp), len(inp[0])
    if not in_c:
        return [row[:] for row in inp]

    all_cells = [(r, c) for col in in_c for r, c in in_c[col]]

    # Find grid offset (r0, c0): cells sit at positions where
    # (r - r0) % 4 in {0,1,2} and (c - c0) % 4 in {0,1,2}
    r0, c0 = None, None
    for r0_ in range(4):
        for c0_ in range(4):
            if all(
                (r - r0_) % 4 in {0, 1, 2} and (c - c0_) % 4 in {0, 1, 2}
                for r, c in all_cells
            ):
                r0, c0 = r0_, c0_
                break
        if r0 is not None:
            break

    def tl(r, c):
        return (r - (r - r0) % 4, c - (c - c0) % 4)

    def meta(t):
        return ((t[0] - r0) // 4, (t[1] - c0) // 4)

    def sgn(x):
        return 1 if x > 0 else (-1 if x < 0 else 0)

    # Group input seeds by (color, meta-position) -> list of template offsets
    seeded: dict = {}
    for color, cells in in_c.items():
        for r, c in cells:
            t = tl(r, c)
            m = meta(t)
            seeded.setdefault(color, {}).setdefault(m, []).append(
                (r - t[0], c - t[1])
            )

    # Find template anchor: the single copy with the most seed cells
    anchor_color, anchor_meta, anchor_poses = None, None, []
    for color, metas in seeded.items():
        for m, poses in metas.items():
            if len(poses) > len(anchor_poses):
                anchor_poses = poses
                anchor_meta = m
                anchor_color = color

    # Determine whether anchor is "complete" (col 0 has all 3 rows seeded)
    anchor_col0 = {r for r, c in anchor_poses if c == 0}
    anchor_complete = 0 in anchor_col0 and 1 in anchor_col0 and 2 in anchor_col0

    # Build the template shape from seed union; apply completion rules only
    # when the anchor is partial (col 0 not fully seeded)
    seed_union: set = set()
    for color, metas in seeded.items():
        for m, poses in metas.items():
            seed_union.update(poses)

    template = set(seed_union)
    if not anchor_complete:
        # Fill col 0 middle row if first and last rows are present
        col0_rows = {r for r, c in template if c == 0}
        if 0 in col0_rows and 2 in col0_rows and 1 not in col0_rows:
            template.add((1, 0))
        # Fill center if (1,0) present and column 1 has any neighbor
        if (1, 0) in template and ((0, 1) in template or (2, 1) in template):
            template.add((1, 1))
    template = frozenset(template)

    def get_step(m, poses, tmeta, is_anchor):
        """Determine the propagation step for a seeded copy at meta m."""
        if is_anchor and anchor_complete:
            return (0, 0)

        vr = m[0] - tmeta[0]
        vc = m[1] - tmeta[1]
        avr, avc = abs(vr), abs(vc)

        n = len(poses)
        avg_dr = sum(r for r, c in poses) / n
        avg_dc = sum(c for r, c in poses) / n
        sdr = sgn(1 - avg_dr)   # +1 leans top row, -1 leans bottom row
        sdc = sgn(1 - avg_dc)   # +1 leans left col, -1 leans right col

        if vr == 0 and vc == 0:
            # At template position: use seed-based direction
            return (sdr, sdc)
        elif vr == 0:
            # Same meta-row as template: row step from seeds,
            # col step from position only if distance > 1
            return (sdr, sgn(vc) if avc > 1 else 0)
        elif vc == 0:
            # Same meta-col as template: row step from position, col from seeds
            return (sgn(vr), sdc)
        elif avr > avc:
            return (sgn(vr), 0)
        elif avc > avr:
            return (0, sgn(vc))
        else:
            return (sgn(vr), sgn(vc))

    # Collect all (color, meta) pairs to render
    copies: dict = {}
    for color, metas in seeded.items():
        seeded_metas = set(metas.keys())
        for m, poses in metas.items():
            is_anc = color == anchor_color and m == anchor_meta
            step = get_step(m, poses, anchor_meta, is_anc)
            copies[(color, m)] = True
            if step == (0, 0):
                continue
            cur = (m[0] + step[0], m[1] + step[1])
            while True:
                t = (r0 + cur[0] * 4, c0 + cur[1] * 4)
                # Stop if no template cell falls within grid bounds
                if not any(
                    0 <= t[0] + dr < H and 0 <= t[1] + dc < W
                    for dr, dc in template
                ):
                    break
                # Stop if we hit another seeded copy of the same color
                if cur in seeded_metas:
                    break
                copies[(color, cur)] = True
                cur = (cur[0] + step[0], cur[1] + step[1])

    # Render output
    out = [[bg] * W for _ in range(H)]
    for (color, m) in copies:
        t = (r0 + m[0] * 4, c0 + m[1] * 4)
        for dr, dc in template:
            r, c = t[0] + dr, t[1] + dc
            if 0 <= r < H and 0 <= c < W:
                out[r][c] = color
    return out
