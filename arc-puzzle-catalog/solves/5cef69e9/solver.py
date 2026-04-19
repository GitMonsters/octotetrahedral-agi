from collections import Counter


def transform(grid):
    grid = [list(row) for row in grid]
    R, C = len(grid), len(grid[0])

    bg = Counter(grid[r][c] for r in range(R) for c in range(C)).most_common(1)[0][0]

    def orbit4(r, c, cr, cc):
        return frozenset([
            (r, c),
            (r, 2 * cc - c),
            (2 * cr - r, c),
            (2 * cr - r, 2 * cc - c),
        ])

    def find_center(cells):
        votes = Counter()
        n = len(cells)
        for i in range(n):
            r1, c1 = cells[i]
            for j in range(i + 1, n):
                r2, c2 = cells[j]
                if grid[r1][c1] == grid[r2][c2]:
                    if (r1 + r2) % 2 == 0 and (c1 + c2) % 2 == 0:
                        votes[((r1 + r2) // 2, (c1 + c2) // 2)] += 1
        if not votes:
            return None
        max_v = max(votes.values())
        cands = [p for p, v in votes.items() if v == max_v]
        # Prefer a non-bg occupied cell as center
        for cr, cc in cands:
            if 0 <= cr < R and 0 <= cc < C and grid[cr][cc] != bg:
                return (cr, cc)
        # Only use a bg-cell center if it has >= 2 votes (strong evidence)
        if max_v >= 2:
            return cands[0]
        return None

    def bfs_clusters(cells):
        """Same-color connections up to distance 4; cross-color only up to distance 2."""
        cells = list(set(cells))
        visited = set()
        clusters = []
        for s in cells:
            if s in visited:
                continue
            cluster = [s]
            visited.add(s)
            q = [s]
            while q:
                r0, c0 = q.pop()
                sc = grid[r0][c0]
                for cell in cells:
                    if cell not in visited:
                        r1, c1 = cell
                        dist = abs(r1 - r0) + abs(c1 - c0)
                        max_d = 4 if grid[r1][c1] == sc else 2
                        if dist <= max_d:
                            visited.add(cell)
                            cluster.append(cell)
                            q.append(cell)
            clusters.append(cluster)
        return clusters

    nongb = [(r, c) for r in range(R) for c in range(C) if grid[r][c] != bg]

    for cluster in bfs_clusters(nongb):
        remaining = cluster[:]
        for _ in range(10):
            if len(remaining) < 2:
                break
            center = find_center(remaining)
            if center is None:
                break
            cr, cc = center

            cell_set = set(remaining)
            seen_orbs = set()
            additions = {}
            orphan_set = set()

            for r, c in remaining:
                orb = orbit4(r, c, cr, cc)
                if orb in seen_orbs:
                    continue
                seen_orbs.add(orb)

                color = grid[r][c]
                present = sum(1 for p in orb if p in cell_set)
                sz = len(orb)
                threshold = (sz + 1) // 2

                if present >= threshold:
                    for pr, pc in orb:
                        if 0 <= pr < R and 0 <= pc < C and grid[pr][pc] == bg:
                            additions[(pr, pc)] = color
                else:
                    for p in orb:
                        if p in cell_set:
                            orphan_set.add(p)

            for (r, c), col in additions.items():
                grid[r][c] = col

            remaining = list(orphan_set)

    return grid
