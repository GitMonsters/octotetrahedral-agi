def transform(input_grid):
    """Complete asymmetric patterns to have 4-fold symmetry.
    
    Each grid has several spatially separated patterns of non-background pixels.
    Some are already symmetric; incomplete ones get completed by reflecting
    existing pixels about the pattern's bounding-box center.
    """
    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    from collections import Counter, defaultdict

    # Background = most common color
    counts = Counter()
    for r in range(rows):
        for c in range(cols):
            counts[grid[r][c]] += 1
    bg = counts.most_common(1)[0][0]

    # Collect non-background pixels
    pixels = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                pixels[(r, c)] = grid[r][c]

    if not pixels:
        return grid

    def do_cluster(pixel_set, dist_fn):
        """Union-find clustering with given distance predicate."""
        parent = {p: p for p in pixel_set}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        plist = list(pixel_set)
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                if dist_fn(plist[i], plist[j]):
                    union(plist[i], plist[j])

        groups = defaultdict(list)
        for p in pixel_set:
            groups[find(p)].append(p)
        return list(groups.values())

    def symmetry_info(cluster):
        """Return (ratio, cr2, cc2) for 4-fold symmetry about bbox center."""
        if len(cluster) <= 1:
            return 1.0, 0, 0

        min_r = min(r for r, c in cluster)
        max_r = max(r for r, c in cluster)
        min_c = min(c for r, c in cluster)
        max_c = max(c for r, c in cluster)

        cr2 = min_r + max_r
        cc2 = min_c + max_c
        cluster_set = set(cluster)

        total = matched = 0
        for r, c in cluster:
            for rr, rc in [(cr2 - r, c), (r, cc2 - c), (cr2 - r, cc2 - c)]:
                total += 1
                if (rr, rc) in cluster_set:
                    matched += 1

        return (matched / total if total else 1.0), cr2, cc2

    def complete_cluster(cluster, cr2, cc2):
        """Fill missing symmetric counterparts."""
        for r, c in cluster:
            color = pixels[(r, c)]
            for rr, rc in [(cr2 - r, c), (r, cc2 - c), (cr2 - r, cc2 - c)]:
                if 0 <= rr < rows and 0 <= rc < cols and grid[rr][rc] == bg:
                    grid[rr][rc] = color

    manhattan_2 = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 2
    chebyshev_1 = lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1

    # Phase 1: broad clustering (Manhattan <= 2)
    for cluster in do_cluster(set(pixels.keys()), manhattan_2):
        if len(cluster) <= 1:
            continue

        ratio, cr2, cc2 = symmetry_info(cluster)

        if 0.5 <= ratio < 1.0:
            complete_cluster(cluster, cr2, cc2)
        elif ratio < 0.5:
            # Phase 2: tighter re-clustering (Chebyshev <= 1)
            for sub in do_cluster(set(cluster), chebyshev_1):
                if len(sub) <= 1:
                    continue
                sub_ratio, sub_cr2, sub_cc2 = symmetry_info(sub)
                if 0.5 <= sub_ratio < 1.0:
                    complete_cluster(sub, sub_cr2, sub_cc2)

    return grid
