def _normalize(cells: frozenset) -> frozenset:
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _rotate90cw(cells: frozenset) -> frozenset:
    """Rotate 90° clockwise: (r,c) → (c, -r), then normalize."""
    return _normalize(frozenset((c, -r) for r, c in cells))


def _three_orientations(cells: frozenset) -> set:
    """Return the set of normalized forms for 0°, 90°CW, and 180° rotations.

    270°CW is excluded because for chirally-asymmetric shapes (e.g. L-shape)
    it equals a horizontal reflection, which would incorrectly match mirror
    images (e.g. J-shape matching an L-room).
    """
    n0 = _normalize(cells)
    n90 = _rotate90cw(n0)
    n180 = _rotate90cw(n90)
    return {n0, n90, n180}


def _bbox(cells):
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _contains_subgraph(blob: frozenset, room_orient: frozenset) -> bool:
    """Return True if room_orient fits inside blob at some offset."""
    rh, rw = _bbox(room_orient)
    bh, bw = _bbox(blob)
    for dr in range(bh - rh + 1):
        for dc in range(bw - rw + 1):
            if frozenset((r + dr, c + dc) for r, c in room_orient).issubset(blob):
                return True
    return False


def _is_pure_column(cells: frozenset) -> bool:
    return len(set(c for _, c in cells)) == 1


def _is_pure_row(cells: frozenset) -> bool:
    return len(set(r for r, _ in cells)) == 1


def _matches_room(blob_cells: list, room_cells: list) -> bool:
    """Return True if blob matches room by the 3-orientation containment rule.

    Rules:
      1. Column room (bbox width=1): blob must be a pure column that contains room.
      2. Row room (bbox height=1): blob must be a pure row that contains room.
      3. General room: blob either exactly equals room in one of 3 orientations,
         or has a strictly larger bounding box and contains room in one of 3 orientations.
    """
    blob = _normalize(frozenset(blob_cells))
    room = frozenset(room_cells)
    bh, bw = _bbox(blob)
    rh, rw = _bbox(room)

    if rw == 1:
        return _is_pure_column(blob) and _contains_subgraph(blob, _normalize(room))
    if rh == 1:
        return _is_pure_row(blob) and _contains_subgraph(blob, _normalize(room))

    for orient in _three_orientations(room):
        if blob == orient:
            return True
        oh, ow = _bbox(orient)
        if bh >= oh and bw >= ow and (bh > oh or bw > ow):
            if _contains_subgraph(blob, orient):
                return True
    return False


def _flood_fill(start_cells: list, passable: set) -> list:
    visited = set()
    comp = set()
    stack = list(start_cells)
    while stack:
        cell = stack.pop()
        if cell in visited or cell not in passable:
            continue
        visited.add(cell)
        comp.add(cell)
        r, c = cell
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if (nr, nc) not in visited and (nr, nc) in passable:
                stack.append((nr, nc))
    return comp


def _find_rooms(grid: list[list[int]]) -> list[list[tuple]]:
    """Find room interiors defined by the 1-structure.

    The cross bar is the unique row where grid[r][0]==1 and grid[r][ncols-1]==1.
    Each connected cluster of 0-cells reachable (4-connected) from a gap in the
    cross bar row, within the 0-cells at or above the cross bar, is one room.
    """
    nrows, ncols = len(grid), len(grid[0])
    cross_row = next(
        (r for r in range(nrows) if grid[r][0] == 1 and grid[r][ncols - 1] == 1),
        None,
    )
    if cross_row is None:
        return []

    zeros_above = {
        (r, c)
        for r in range(cross_row + 1)
        for c in range(ncols)
        if grid[r][c] == 0
    }
    gap_cells = [(cross_row, c) for c in range(ncols) if grid[cross_row][c] == 0]

    rooms = []
    visited_gaps = set()
    for start in gap_cells:
        if start in visited_gaps:
            continue
        comp = _flood_fill([start], zeros_above)
        for cell in comp:
            visited_gaps.add(cell)
        rooms.append(list(comp))
    return rooms


def _connected_components(cells: list, color_val: int, grid: list[list[int]]) -> list:
    nrows, ncols = len(grid), len(grid[0])
    cell_set = set(cells)
    visited = set()
    comps = []
    for cell in cells:
        if cell in visited:
            continue
        comp = set()
        stack = [cell]
        while stack:
            cur = stack.pop()
            if cur in visited or cur not in cell_set:
                continue
            visited.add(cur)
            comp.add(cur)
            r, c = cur
            for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if (nr, nc) not in visited and (nr, nc) in cell_set:
                    stack.append((nr, nc))
        comps.append(list(comp))
    return comps


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Solve ARC task 1acc24af.

    Rule: The 1-structure defines rooms (enclosed 0-cell shapes inside the box bumps).
    Each 5-blob is compared against every room using a 3-orientation containment rule.
    If the blob matches any room it is recolored 2; otherwise it stays 5.

    Matching rule for a blob B vs room R:
      - Column room (R width=1): B must be a pure column containing R as a sub-run.
      - Row room (R height=1): B must be a pure row containing R as a sub-run.
      - General room: B equals R in one of {0°, 90°CW, 180°} orientations (exact),
        OR B has a strictly larger bounding box and contains R in one of those orientations.
      270°CW rotation is excluded to avoid treating a shape's mirror image as a match.
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    rooms = _find_rooms(grid)
    if not rooms:
        return result

    five_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
    blobs = _connected_components(five_cells, 5, grid)

    for blob in blobs:
        for room in rooms:
            if _matches_room(blob, room):
                for r, c in blob:
                    result[r][c] = 2
                break

    return result


if __name__ == "__main__":
    import json

    with open(
        "/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/1acc24af.json"
    ) as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Test {i}: {status}")
