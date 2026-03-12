from collections import deque


def get_components(cells):
    """Get connected components (4-connected) from a list of cells."""
    cell_set = set(cells)
    visited = set()
    components = []
    for cell in sorted(cells):
        if cell not in visited:
            comp = []
            q = deque([cell])
            visited.add(cell)
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) in cell_set and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))
            components.append(sorted(comp))
    return components


def normalize_shape(cells):
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def all_orientations(shape):
    """Generate all 8 orientations (4 rotations x 2 reflections)."""
    variants = set()
    cells = list(shape)
    for _ in range(4):
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        norm = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
        variants.add(norm)
        max_c = max(c for r, c in cells)
        mirror = [(r, max_c - c) for r, c in cells]
        min_r2 = min(r for r, c in mirror)
        min_c2 = min(c for r, c in mirror)
        norm_m = tuple(sorted((r - min_r2, c - min_c2) for r, c in mirror))
        variants.add(norm_m)
        cells = [(c, -r) for r, c in cells]  # rotate 90° CW
    return variants


def shape_contains(blob_cells, room_shape):
    """Check if blob contains room shape (any orientation) as a connected sub-pattern."""
    blob_set = set(blob_cells)
    for variant in all_orientations(room_shape):
        for br, bc in blob_cells:
            for vr, vc in variant:
                offset_r, offset_c = br - vr, bc - vc
                translated = {(vr2 + offset_r, vc2 + offset_c) for vr2, vc2 in variant}
                if translated.issubset(blob_set):
                    return True
    return False


def find_rooms(grid):
    """Find interior 0-cells of the 1-structure.

    A cell is interior if it is 0, lies between 1s on the same row,
    and has at least one 1 above it in the same column.
    """
    rows, cols = len(grid), len(grid[0])
    one_rows = [r for r in range(rows) if any(grid[r][c] == 1 for c in range(cols))]
    interior = []
    for r in one_rows:
        for c in range(cols):
            if grid[r][c] != 0:
                continue
            has_left = any(grid[r][cc] == 1 for cc in range(c))
            has_right = any(grid[r][cc] == 1 for cc in range(c + 1, cols))
            if not (has_left and has_right):
                continue
            has_above = any(grid[rr][c] == 1 for rr in range(r))
            if has_above:
                interior.append((r, c))
    return get_components(interior)


def col_distance(blob_cols, room_cols):
    return min(abs(bc - rc) for bc in blob_cols for rc in room_cols)


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Solve ARC task 1acc24af.

    Rule: The 1-structure in the upper portion defines rooms (enclosed 0-cells).
    Each 5-blob in the lower portion is matched to its nearest room (by column
    distance). If the blob's shape contains the room's shape (in any orientation)
    as a connected sub-pattern, the blob is recolored to 2. Otherwise it stays 5.
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    rooms = find_rooms(grid)
    if not rooms:
        return result

    room_shapes = [normalize_shape(room) for room in rooms]
    room_col_sets = [set(c for _, c in room) for room in rooms]

    five_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
    five_comps = get_components(five_cells)

    for comp in five_comps:
        blob_cols = set(c for _, c in comp)

        # Find nearest room by column distance
        min_dist = float("inf")
        nearest_idx = 0
        for ri, rcols in enumerate(room_col_sets):
            d = col_distance(blob_cols, rcols)
            if d < min_dist:
                min_dist = d
                nearest_idx = ri

        if shape_contains(comp, room_shapes[nearest_idx]):
            for r, c in comp:
                result[r][c] = 2

    return result


if __name__ == "__main__":
    import json

    with open(
        "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/1acc24af.json"
    ) as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
