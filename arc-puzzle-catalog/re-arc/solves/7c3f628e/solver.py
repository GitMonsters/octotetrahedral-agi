def transform(grid):
    from collections import Counter
    H = len(grid); W = len(grid[0])
    cc = Counter()
    for row in grid: cc.update(row)
    bg = cc.most_common(1)[0][0]
    markers = [(r, c, grid[r][c]) for r in range(H) for c in range(W) if grid[r][c] != bg]

    # Compute max_ring for each marker using collinear-pair restrictions.
    # When two markers share a column, the one with the smaller row index is restricted
    # to even rings up to floor(row_distance/2), rounded down to the nearest even.
    # Same rule applies for shared rows (restrict smaller column index).
    max_ring = {(mr, mc): float('inf') for mr, mc, mv in markers}
    for i, (r1, c1, v1) in enumerate(markers):
        for j, (r2, c2, v2) in enumerate(markers):
            if i >= j: continue
            if c1 == c2:  # same column: restrict smaller-row marker
                dist = abs(r1 - r2)
                restriction = (dist // 2) & ~1  # largest even <= dist//2
                key = (r1, c1) if r1 < r2 else (r2, c2)
                max_ring[key] = min(max_ring[key], restriction)
            elif r1 == r2:  # same row: restrict smaller-col marker
                dist = abs(c1 - c2)
                restriction = (dist // 2) & ~1
                key = (r1, c1) if c1 < c2 else (r2, c2)
                max_ring[key] = min(max_ring[key], restriction)

    result = [[bg] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            dists = [
                (max(abs(r - mr), abs(c - mc)), abs(r - mr) + abs(c - mc), mr, mc, mv)
                for mr, mc, mv in markers
            ]
            dists.sort()
            d0 = dists[0][0]
            if d0 % 2 == 1:
                continue  # odd Chebyshev ring -> background

            tied = [x for x in dists if x[0] == d0]

            if len(tied) > 1:
                # Filter to markers whose ring limit allows coloring at d0
                unblocked = [x for x in tied if d0 <= max_ring[(x[2], x[3])]]
                if len(unblocked) <= 1:
                    continue  # equidistant with <=1 viable candidate -> background
                # Multiple viable candidates: use Manhattan distance as tiebreaker
                best = min(unblocked, key=lambda x: x[1])
                result[r][c] = best[4]
            else:
                mr0, mc0, mv0 = dists[0][2], dists[0][3], dists[0][4]
                if d0 <= max_ring[(mr0, mc0)]:
                    result[r][c] = mv0

    return result
