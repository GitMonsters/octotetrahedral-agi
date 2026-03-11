"""
ARC-AGI puzzle 9385bd28 solver.

Pattern: Pairs of L-shaped triominoes define rectangles. A legend maps
object colors to fill colors. Each pair's bounding box is filled. Fills
are layered by legend position (topmost legend entry = highest priority).
Non-legend L-pairs have their cells and missing corners protected.
If a non-legend L-pair's bbox is fully inside an active fill, the whole
bbox acts as a transparent hole.
"""

import json
from collections import defaultdict


def transform(grid):
    H = len(grid)
    W = len(grid[0])

    counts = defaultdict(int)
    for row in grid:
        for v in row:
            counts[v] += 1
    bg = max(counts, key=counts.get)

    color_pos = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                color_pos[grid[r][c]].append((r, c))

    def find_clusters(positions):
        pos_set = set(positions)
        visited = set()
        clusters = []
        for p in positions:
            if p in visited:
                continue
            cluster = []
            stack = [p]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                cluster.append(cur)
                r, c = cur
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = (r + dr, c + dc)
                    if nb in pos_set and nb not in visited:
                        stack.append(nb)
            clusters.append(sorted(cluster))
        return clusters

    def get_missing_corner(cluster):
        rs = [r for r, c in cluster]
        cs = [c for r, c in cluster]
        s = set(cluster)
        for r in range(min(rs), max(rs) + 1):
            for c in range(min(cs), max(cs) + 1):
                if (r, c) not in s:
                    return (r, c)
        return None

    def bbox_of(positions):
        rs = [r for r, c in positions]
        cs = [c for r, c in positions]
        return (min(rs), max(rs), min(cs), max(cs))

    def bbox_contains(outer, inner):
        return (outer[0] <= inner[0] and inner[1] <= outer[1] and
                outer[2] <= inner[2] and inner[3] <= outer[3])

    def cell_in_bbox(r, c, bbox):
        return bbox[0] <= r <= bbox[1] and bbox[2] <= c <= bbox[3]

    # Classify clusters
    l_pairs = {}
    legend_candidates = []
    for color in sorted(color_pos.keys()):
        clusters = find_clusters(color_pos[color])
        size3 = [c for c in clusters if len(c) == 3]
        small = [c for c in clusters if len(c) <= 2]
        if len(size3) == 2:
            l_pairs[color] = (size3[0], size3[1])
        for cl in small:
            for pos in cl:
                legend_candidates.append((pos, color))

    # Parse legend (horizontal adjacent pairs, left=source, right=fill)
    legend_positions = {pos: color for pos, color in legend_candidates}
    legend_mapping = {}
    legend_rows = {}
    used = set()
    for (r, c), color in sorted(legend_candidates):
        if (r, c) in used:
            continue
        right = (r, c + 1)
        if right in legend_positions and right not in used:
            legend_mapping[color] = legend_positions[right]
            legend_rows[color] = r
            used.add((r, c))
            used.add(right)

    legend_cell_set = {pos for pos, _ in legend_candidates}

    # Active L-pair fills (source has L-pair and fill != 0)
    active_fills = {}
    for src, fill in legend_mapping.items():
        if fill == 0:
            continue
        if src in l_pairs:
            active_fills[src] = fill

    # Non-L-pair fills (source in legend, no L-pair, fill != 0)
    non_lpair_fills = {}
    for src, fill in legend_mapping.items():
        if fill == 0 or src in l_pairs:
            continue
        non_legend = [p for p in color_pos.get(src, []) if p not in legend_cell_set]
        if non_legend:
            non_lpair_fills[src] = (fill, non_legend)

    # Compute bboxes and fill colors for all fills
    bboxes = {}
    fill_colors = {}
    for color in active_fills:
        L1, L2 = l_pairs[color]
        bboxes[color] = bbox_of(L1 + L2)
        fill_colors[color] = active_fills[color]
    for color in non_lpair_fills:
        fill, positions = non_lpair_fills[color]
        bboxes[color] = bbox_of(positions)
        fill_colors[color] = fill

    # Non-legend L-pairs
    non_legend_colors = set()
    non_legend_lpair_bboxes = {}
    for color in l_pairs:
        if color not in legend_mapping:
            non_legend_colors.add(color)
            L1, L2 = l_pairs[color]
            non_legend_lpair_bboxes[color] = bbox_of(L1 + L2)

    # Holes: non-legend L-pair bbox fully inside a fill's bbox
    holes_per_fill = defaultdict(list)
    for nl_color, nl_bbox in non_legend_lpair_bboxes.items():
        for fill_src, fill_bbox in bboxes.items():
            if bbox_contains(fill_bbox, nl_bbox):
                holes_per_fill[fill_src].append(nl_bbox)

    # Protected cells: non-legend L-shape cells + their missing corners
    protected_cells = set()
    for color in non_legend_colors:
        L1, L2 = l_pairs[color]
        for cell in L1 + L2:
            protected_cells.add(cell)
        for cl in (L1, L2):
            mc = get_missing_corner(cl)
            if mc:
                protected_cells.add(mc)

    # Sort by legend row descending (bottom = lowest priority, top = highest)
    sorted_fills = sorted(
        fill_colors.keys(),
        key=lambda c: legend_rows.get(c, 0),
        reverse=True,
    )

    result = [row[:] for row in grid]

    # Apply fills in priority order
    for color in sorted_fills:
        fc = fill_colors[color]
        r0, r1, c0, c1 = bboxes[color]
        holes = holes_per_fill.get(color, [])
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if any(cell_in_bbox(r, c, h) for h in holes):
                    continue
                if (r, c) in protected_cells:
                    continue
                result[r][c] = fc

    # Restore active L-shape cells on top
    for color in active_fills:
        L1, L2 = l_pairs[color]
        for r, c in L1 + L2:
            result[r][c] = color

    return result


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/9385bd28.json"
    )
    with open(path) as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = transform(ex["input"])
        ok = result == ex["output"]
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != ex["output"][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]} expected {ex['output'][r][c]}")

    for i, ex in enumerate(data.get("test", [])):
        result = transform(ex["input"])
        if "output" in ex:
            ok = result == ex["output"]
            print(f"Test  {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
                for r in range(len(result)):
                    for c in range(len(result[0])):
                        if result[r][c] != ex["output"][r][c]:
                            print(f"  ({r},{c}): got {result[r][c]} expected {ex['output'][r][c]}")
        else:
            print(f"Test  {i}: output generated")
            for row in result:
                print("".join(str(v) for v in row))

    if all_pass:
        print("\nAll examples passed!")
