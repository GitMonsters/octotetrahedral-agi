def solve(grid: list[list[int]]) -> list[list[int]]:
    """Solve ARC task 18419cfa.
    
    Rule: Each 8-frame (box) contains a small 2-pattern in one half.
    The frame has notch-like protrusions indicating the reflection axis.
    - Notches on top/bottom (fewer 8s on top/bottom edges) → vertical reflection
    - Notches on left/right (fewer 8s on left/right edges) → horizontal reflection
    The 2-pattern is reflected to fill the opposite half of the interior.
    """
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find connected components of 8s (frames)
    visited = [[False] * cols for _ in range(rows)]
    frames: list[set[tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and not visited[r][c]:
                comp: set[tuple[int, int]] = set()
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 8:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                frames.append(comp)

    # Find exterior cells (reachable from grid border without crossing 8s)
    exterior: set[tuple[int, int]] = set()
    queue: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid[r][c] != 8:
                if (r, c) not in exterior:
                    exterior.add((r, c))
                    queue.append((r, c))
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in exterior and grid[nr][nc] != 8:
                exterior.add((nr, nc))
                queue.append((nr, nc))

    # All interior cells (non-8, non-exterior)
    all_interior: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 8 and (r, c) not in exterior:
                all_interior.add((r, c))

    # Process each frame
    for frame_set in frames:
        min_r = min(r for r, c in frame_set)
        max_r = max(r for r, c in frame_set)
        min_c = min(c for r, c in frame_set)
        max_c = max(c for r, c in frame_set)

        # Interior cells for this frame
        interior = {(r, c) for r in range(min_r, max_r + 1)
                    for c in range(min_c, max_c + 1)
                    if (r, c) in all_interior}

        if not interior:
            continue

        two_cells = [(r, c) for r, c in interior if grid[r][c] == 2]
        if not two_cells:
            continue

        # Detect notch direction from edge counts
        top_count = sum(1 for c in range(min_c, max_c + 1) if (min_r, c) in frame_set)
        bot_count = sum(1 for c in range(min_c, max_c + 1) if (max_r, c) in frame_set)
        left_count = sum(1 for r in range(min_r, max_r + 1) if (r, min_c) in frame_set)
        right_count = sum(1 for r in range(min_r, max_r + 1) if (r, max_c) in frame_set)

        tb_count = top_count + bot_count
        lr_count = left_count + right_count

        int_min_r = min(r for r, c in interior)
        int_max_r = max(r for r, c in interior)
        int_min_c = min(c for r, c in interior)
        int_max_c = max(c for r, c in interior)

        if tb_count < lr_count:
            # Notches on top/bottom → vertical (top↔bottom) reflection
            for r, c in two_cells:
                ref_r = int_min_r + int_max_r - r
                if (ref_r, c) in interior and result[ref_r][c] != 2:
                    result[ref_r][c] = 2
        else:
            # Notches on left/right → horizontal (left↔right) reflection
            for r, c in two_cells:
                ref_c = int_min_c + int_max_c - c
                if (r, ref_c) in interior and result[r][ref_c] != 2:
                    result[r][ref_c] = 2

    return result


if __name__ == "__main__":
    import json, sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/18419cfa.json"
    with open(path) as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        if "output" in ex:
            status = "PASS" if result == ex["output"] else "FAIL"
            print(f"Test {i}: {status}")
