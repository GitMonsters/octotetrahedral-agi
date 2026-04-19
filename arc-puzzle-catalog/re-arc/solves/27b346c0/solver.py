from collections import Counter
from itertools import product


def cluster_cells_constrained(cells):
    """Cluster non-bg cells so each cluster fits within a 3x3 bounding box."""
    n = len(cells)
    if n <= 1:
        return [cells]

    dists = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = ((cells[i][0] - cells[j][0]) ** 2 + (cells[i][1] - cells[j][1]) ** 2) ** 0.5
            dists[i][j] = d
            dists[j][i] = d

    cluster_ids = list(range(n))
    cluster_members = {i: [i] for i in range(n)}

    pairs = sorted((dists[i][j], i, j) for i in range(n) for j in range(i + 1, n))

    def get_bbox(members):
        rs = [cells[m][0] for m in members]
        cs = [cells[m][1] for m in members]
        return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    def find(x):
        while cluster_ids[x] != x:
            cluster_ids[x] = cluster_ids[cluster_ids[x]]
            x = cluster_ids[x]
        return x

    for d, i, j in pairs:
        ci, cj = find(i), find(j)
        if ci == cj:
            continue
        merged = cluster_members[ci] + cluster_members[cj]
        h, w = get_bbox(merged)
        if h <= 3 and w <= 3:
            cluster_ids[ci] = cj
            cluster_members[cj] = merged
            del cluster_members[ci]

    result = {}
    for i in range(n):
        root = find(i)
        result.setdefault(root, []).append(i)
    return [[cells[i] for i in members] for members in result.values()]


def get_alignments(group):
    """Enumerate all valid 3x3 alignments for a group of cells."""
    rs = [r for r, c, v in group]
    cs = [c for r, c, v in group]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    alignments = []
    for sr in range(max_r - 2, min_r + 1):
        for sc in range(max_c - 2, min_c + 1):
            valid = all(0 <= r - sr <= 2 and 0 <= c - sc <= 2 for r, c, v in group)
            if valid:
                mapping = {}
                for r, c, v in group:
                    mapping[(r - sr, c - sc)] = v
                alignments.append(mapping)
    return alignments


def transform(grid):
    R, C = len(grid), len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    cells = [(r, c, grid[r][c]) for r in range(R) for c in range(C) if grid[r][c] != bg]

    groups = cluster_cells_constrained(cells)
    group_alignments = [get_alignments(g) for g in groups]

    best_result = None
    best_overlaps = -1

    for combo in product(*group_alignments):
        merged = {}
        conflict = False
        for mapping in combo:
            for pos, v in mapping.items():
                if pos in merged and merged[pos] != v:
                    conflict = True
                    break
                merged[pos] = v
            if conflict:
                break
        if conflict:
            continue

        used_count = {}
        for mapping in combo:
            for pos in mapping:
                used_count[pos] = used_count.get(pos, 0) + 1
        overlaps = sum(v - 1 for v in used_count.values() if v > 1)

        if overlaps > best_overlaps:
            best_overlaps = overlaps
            result = [[bg] * 3 for _ in range(3)]
            for (r, c), v in merged.items():
                result[r][c] = v
            best_result = result

    return best_result
