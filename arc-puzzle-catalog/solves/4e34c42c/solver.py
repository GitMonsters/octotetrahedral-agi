import json
import sys
from collections import Counter, deque


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find connected components of non-bg cells (4-connected)
    visited = [[False] * cols for _ in range(rows)]
    raw_components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                comp = {}
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp[(cr, cc)] = grid[cr][cc]
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                raw_components.append(comp)

    def normalize(comp):
        min_r = min(r for r, c in comp)
        min_c = min(c for r, c in comp)
        return {(r - min_r, c - min_c): v for (r, c), v in comp.items()}

    components = [normalize(c) for c in raw_components]

    def is_subset(a, b):
        if len(a) > len(b):
            return False
        a_cells = list(a.items())
        if not a_cells:
            return True
        (r0, c0), v0 = a_cells[0]
        for (rb, cb), vb in b.items():
            if vb == v0:
                dr, dc = rb - r0, cb - c0
                if all(b.get((r + dr, c + dc)) == v for (r, c), v in a_cells):
                    return True
        return False

    # Remove subsets
    keep = []
    for i in range(len(components)):
        contained = False
        for j in range(len(components)):
            if i != j and len(components[j]) >= len(components[i]):
                if is_subset(components[i], components[j]):
                    contained = True
                    break
        if not contained:
            keep.append(components[i])
    components = keep
    n = len(components)

    if n == 1:
        comp = components[0]
        max_r = max(r for r, c in comp)
        max_c = max(c for r, c in comp)
        result = [[bg] * (max_c + 1) for _ in range(max_r + 1)]
        for (r, c), v in comp.items():
            result[r][c] = v
        return result

    def comp_bbox(comp):
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        return (min(rs), max(rs), min(cs), max(cs))

    comp_bboxes = [comp_bbox(c) for c in components]

    # Color counts per component for specificity scoring
    comp_color_counts = [Counter(comp.values()) for comp in components]

    def find_valid_offsets(a, b):
        candidates = set()
        for (ra, ca), va in a.items():
            for (rb, cb), vb in b.items():
                if va == vb:
                    candidates.add((ra - rb, ca - cb))
        valid = []
        for dr, dc in candidates:
            overlap = 0
            conflict = False
            new_cells = 0
            for (rb, cb), vb in b.items():
                ar, ac = rb + dr, cb + dc
                if (ar, ac) in a:
                    if a[(ar, ac)] != vb:
                        conflict = True
                        break
                    overlap += 1
                else:
                    new_cells += 1
            if not conflict and overlap > 0 and new_cells > 0:
                valid.append((overlap, dr, dc))
        valid.sort(key=lambda x: -x[0])
        return valid

    pair_offsets = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                offsets = find_valid_offsets(components[i], components[j])
                if offsets:
                    pair_offsets[(i, j)] = offsets


    def _build(cell_map, bbox, bg):
        min_r, max_r, min_c, max_c = bbox
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        result = [[bg] * w for _ in range(h)]
        for (r, c), v in cell_map.items():
            result[r - min_r][c - min_c] = v
        return result

    # Search: maximize total quality, lex-largest output as tiebreaker
    best_solution = [None, -float('inf'), None]  # cell_map, total_quality, output

    def search(placed_set, positions, cell_map, bbox, total_quality):
        if len(placed_set) == n:
            if total_quality > best_solution[1]:
                best_solution[0] = dict(cell_map)
                best_solution[1] = total_quality
                best_solution[2] = _build(cell_map, bbox, bg)
            elif total_quality == best_solution[1]:
                out = _build(cell_map, bbox, bg)
                if best_solution[2] is None or out > best_solution[2]:
                    best_solution[0] = dict(cell_map)
                    best_solution[1] = total_quality
                    best_solution[2] = out
            return

        candidates = []
        for ni in range(n):
            if ni in placed_set:
                continue
            ni_minr, ni_maxr, ni_minc, ni_maxc = comp_bboxes[ni]
            len_ni = len(components[ni])
            for pi in placed_set:
                key = (pi, ni)
                if key not in pair_offsets:
                    continue
                len_pi = len(components[pi])
                for overlap, dr, dc in pair_offsets[key]:
                    gdr = positions[pi][0] + dr
                    gdc = positions[pi][1] + dc

                    ok = True
                    olap = 0
                    new = 0
                    specificity = 0.0
                    for (rb, cb), vb in components[ni].items():
                        gr, gc = rb + gdr, cb + gdc
                        if (gr, gc) in cell_map:
                            if cell_map[(gr, gc)] != vb:
                                ok = False
                                break
                            olap += 1
                            frac_pi_c = comp_color_counts[pi][vb] / len_pi
                            frac_ni_c = comp_color_counts[ni][vb] / len_ni
                            specificity += (1 - max(frac_pi_c, frac_ni_c)) ** 2
                        else:
                            new += 1

                    if ok and olap > 0 and new > 0:
                        quality = specificity + olap * 0.0001
                        new_bbox = (
                            min(bbox[0], ni_minr + gdr),
                            max(bbox[1], ni_maxr + gdr),
                            min(bbox[2], ni_minc + gdc),
                            max(bbox[3], ni_maxc + gdc),
                        )
                        candidates.append((quality, olap, new, ni, gdr, gdc, new_bbox))

        # Sort by quality descending
        candidates.sort(key=lambda x: (-x[0], -x[1]))

        # Deduplicate: same (ni, gdr, gdc) keep best
        seen = set()
        unique = []
        for entry in candidates:
            key = (entry[3], entry[4], entry[5])
            if key not in seen:
                seen.add(key)
                unique.append(entry)

        # Limit candidates to top 20 per step
        for quality, olap, new, ni, gdr, gdc, new_bbox in unique[:20]:
            new_positions = dict(positions)
            new_positions[ni] = (gdr, gdc)
            new_cell_map = dict(cell_map)
            for (rb, cb), vb in components[ni].items():
                new_cell_map[(rb + gdr, cb + gdc)] = vb
            new_placed = placed_set | {ni}
            search(new_placed, new_positions, new_cell_map, new_bbox,
                   total_quality + quality)

    for start in range(n):
        bb = comp_bboxes[start]
        search({start}, {start: (0, 0)}, dict(components[start]), bb, 0.0)

    if best_solution[0] is None:
        largest = max(components, key=len)
        max_r = max(r for r, c in largest)
        max_c = max(c for r, c in largest)
        result = [[bg] * (max_c + 1) for _ in range(max_r + 1)]
        for (r, c), v in largest.items():
            result[r][c] = v
        return result

    cell_map = best_solution[0]
    min_r = min(r for r, c in cell_map)
    max_r = max(r for r, c in cell_map)
    min_c = min(c for r, c in cell_map)
    max_c = max(c for r, c in cell_map)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    result = [[bg] * w for _ in range(h)]
    for (r, c), v in cell_map.items():
        result[r - min_r][c - min_c] = v
    return result


if __name__ == "__main__":
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/4e34c42c.json") as f:
        data = json.load(f)

    all_pass = True
    for i, example in enumerate(data["train"]):
        result = transform(example["input"])
        expected = example["output"]
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            all_pass = False
            print(f"Train {i}: FAIL")
            print(f"  Expected size: {len(expected)}x{len(expected[0])}")
            print(f"  Got size:      {len(result)}x{len(result[0])}")
            for r in range(max(len(expected), len(result))):
                exp_row = expected[r] if r < len(expected) else None
                got_row = result[r] if r < len(result) else None
                if exp_row != got_row:
                    print(f"  Row {r} exp: {exp_row}")
                    print(f"  Row {r} got: {got_row}")

    if all_pass:
        print("\nAll training examples passed!")

    print("\nTest predictions:")
    for i, example in enumerate(data["test"]):
        result = transform(example["input"])
        print(f"\nTest {i} ({len(result)}x{len(result[0])}):")
        for row in result:
            print(row)
