from collections import Counter


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    H_in = len(input_grid)
    W_in = len(input_grid[0])

    # Find background (most common value)
    vals = Counter(v for row in input_grid for v in row)
    bg = vals.most_common(1)[0][0]

    # Find corner markers: 4 same-colored dots forming a rectangle
    dots_by_val: dict[int, list[tuple[int, int]]] = {}
    for r in range(H_in):
        for c in range(W_in):
            if input_grid[r][c] != bg:
                dots_by_val.setdefault(input_grid[r][c], []).append((r, c))

    corner_val = None
    r1 = r2 = c1 = c2 = 0
    for v, pts in dots_by_val.items():
        if len(pts) >= 4:
            rows = sorted(set(r for r, c in pts))
            cols = sorted(set(c for r, c in pts))
            if len(rows) == 2 and len(cols) == 2:
                corners = {(r, c) for r in rows for c in cols}
                if corners <= set(pts):
                    corner_val, r1, r2, c1, c2 = v, rows[0], rows[1], cols[0], cols[1]
                    break

    H = r2 - r1 + 1
    W = c2 - c1 + 1

    # Build output grid with bg and corners
    output = [[bg] * W for _ in range(H)]
    output[0][0] = corner_val
    output[0][W - 1] = corner_val
    output[H - 1][0] = corner_val
    output[H - 1][W - 1] = corner_val

    # For each non-corner color group, find shift(s) and place dots
    for v, all_pts in dots_by_val.items():
        if v == corner_val:
            continue
        dots = all_pts

        shifts = _find_shifts(dots, H, W)
        if len(shifts) == 1:
            dr, dc = shifts[0]
            for r, c in dots:
                output[r - dr][c - dc] = v
        elif len(shifts) > 1:
            # Multiple valid shifts — pick one that avoids corner collisions
            for dr, dc in shifts:
                collision = False
                for r, c in dots:
                    or_, oc = r - dr, c - dc
                    if (or_, oc) in {(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)}:
                        collision = True
                        break
                if not collision:
                    for r, c in dots:
                        output[r - dr][c - dc] = v
                    break
            else:
                dr, dc = shifts[0]
                for r, c in dots:
                    output[r - dr][c - dc] = v
        else:
            # No single shift covers all dots — use greedy set cover
            cover = _greedy_set_cover(dots, H, W)
            for (dr, dc), covered_indices in cover:
                for i in covered_indices:
                    r, c = dots[i]
                    output[r - dr][c - dc] = v

    return output


def _find_shifts(dots: list[tuple[int, int]], H: int, W: int) -> list[tuple[int, int]]:
    """Find all (dr,dc) that place every dot on the H×W rectangle perimeter."""
    dr_cands = set()
    dc_cands = set()
    for r, c in dots:
        dr_cands.add(r)
        dr_cands.add(r - H + 1)
        dc_cands.add(c)
        dc_cands.add(c - W + 1)

    results = []
    for dr in dr_cands:
        for dc in dc_cands:
            if all(_on_perim(r - dr, c - dc, H, W) for r, c in dots):
                results.append((dr, dc))
    return results


def _on_perim(or_: int, oc: int, H: int, W: int) -> bool:
    """Check if (or_, oc) is on the perimeter of an H×W rectangle."""
    return (0 <= or_ <= H - 1 and 0 <= oc <= W - 1 and
            (or_ in (0, H - 1) or oc in (0, W - 1)))


def _greedy_set_cover(
    dots: list[tuple[int, int]], H: int, W: int
) -> list[tuple[tuple[int, int], set[int]]]:
    """Find minimum shifts to cover all dots via greedy set cover."""
    # Generate candidate shifts from each dot's valid placements
    candidates: dict[tuple[int, int], set[int]] = {}
    for i, (r, c) in enumerate(dots):
        for dr in range(r - H + 1, r + 1):
            for dc_val in (c, c - W + 1):
                or_, oc = r - dr, c - dc_val
                if _on_perim(or_, oc, H, W):
                    key = (dr, dc_val)
                    candidates.setdefault(key, set()).add(i)
            for dc in range(c - W + 1, c + 1):
                or_, oc = r - dr, c - dc
                if _on_perim(or_, oc, H, W):
                    key = (dr, dc)
                    candidates.setdefault(key, set()).add(i)

    # Verify each candidate shift against ALL dots it claims to cover
    verified: dict[tuple[int, int], set[int]] = {}
    for shift, indices in candidates.items():
        dr, dc = shift
        valid = set()
        for i in indices:
            r, c = dots[i]
            if _on_perim(r - dr, c - dc, H, W):
                valid.add(i)
        if valid:
            verified[shift] = valid

    # Greedy set cover
    all_indices = set(range(len(dots)))
    uncovered = set(all_indices)
    selected: list[tuple[tuple[int, int], set[int]]] = []

    while uncovered:
        best_shift = None
        best_covered: set[int] = set()
        for shift, covered in verified.items():
            overlap = covered & uncovered
            if len(overlap) > len(best_covered):
                best_covered = overlap
                best_shift = shift
        if best_shift is None:
            break
        selected.append((best_shift, best_covered))
        uncovered -= best_covered

    return selected
