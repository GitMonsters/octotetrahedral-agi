def transform(input_grid: list[list[int]]) -> list[list[int]]:
    """Each gray shape in the bottom half matches a hole pattern in the red region.
    The shape is placed at the hole position (aligned by matching its top rows to
    the hole shape), colored blue(1). Gray is removed."""
    rows = len(input_grid)
    cols = len(input_grid[0])
    output = [row[:] for row in input_grid]

    # Find red_bottom (first all-0 row)
    red_bottom = 0
    for r in range(rows):
        if all(v == 0 for v in input_grid[r]):
            red_bottom = r
            break

    # Find holes in red region and gray cells
    holes = set()
    gray_cells = set()
    for r in range(rows):
        for c in range(cols):
            if r < red_bottom and input_grid[r][c] == 0:
                holes.add((r, c))
            if input_grid[r][c] == 5:
                gray_cells.add((r, c))

    # BFS connected components
    def find_components(cells):
        remaining = set(cells)
        components = []
        while remaining:
            start = next(iter(remaining))
            component = set()
            queue = [start]
            while queue:
                cell = queue.pop(0)
                if cell in remaining:
                    remaining.discard(cell)
                    component.add(cell)
                    r, c = cell
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if (r + dr, c + dc) in remaining:
                            queue.append((r + dr, c + dc))
            components.append(component)
        return components

    def normalize(cells):
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        return frozenset((r - min_r, c - min_c) for r, c in cells), min_r, min_c

    hole_groups = find_components(holes)
    gray_shapes = find_components(gray_cells)

    # Remove gray from output
    for r, c in gray_cells:
        output[r][c] = 0

    # Normalize gray shapes
    gray_norms = [normalize(gs)[0] for gs in gray_shapes]

    used_gray = set()

    for hg in hole_groups:
        h_norm, h_min_r, h_min_c = normalize(hg)
        h_height = max(r for r, c in h_norm) + 1

        for gi, g_norm in enumerate(gray_norms):
            if gi in used_gray:
                continue
            # Extract top h_height rows of gray shape
            g_top = frozenset((r, c) for r, c in g_norm if r < h_height)
            if not g_top:
                continue
            # Normalize columns
            g_top_min_c = min(c for r, c in g_top)
            g_top_norm = frozenset((r, c - g_top_min_c) for r, c in g_top)

            if g_top_norm == h_norm:
                used_gray.add(gi)
                start_r = h_min_r
                start_c = h_min_c - g_top_min_c
                for dr, dc in g_norm:
                    pr, pc = start_r + dr, start_c + dc
                    if 0 <= pr < rows and 0 <= pc < cols and output[pr][pc] == 0:
                        output[pr][pc] = 1
                break

    return output


if __name__ == "__main__":
    ex0_in = [[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],[2,2,2,2,0,2,2,2,0,2,2,0,0,2,2],[2,0,0,2,0,2,2,0,0,0,2,0,0,2,2],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,5,5,0,0,0,0,0,0,0,0,0,0,0,0],[0,5,5,0,0,0,0,0,0,0,5,0,0,0,0],[5,5,5,5,0,0,0,5,0,0,5,0,0,5,5],[0,5,5,0,0,0,5,5,5,0,5,0,5,5,5]]
    ex0_out = [[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],[2,2,2,2,1,2,2,2,1,2,2,1,1,2,2],[2,1,1,2,1,2,2,1,1,1,2,1,1,2,2],[1,1,1,0,1,0,0,0,0,0,1,1,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    ex1_in = [[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],[2,0,2,2,2,2,2,2,2,2,2,2,2,2,0],[2,0,0,2,2,2,0,0,0,2,2,2,2,2,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,5,0],[0,0,0,0,0,0,5,0,0,0,0,0,0,5,0],[0,5,5,5,0,0,5,5,0,0,0,0,0,5,0],[0,5,5,5,0,0,5,5,5,0,0,0,0,5,0]]
    ex1_out = [[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],[2,1,2,2,2,2,2,2,2,2,2,2,2,2,1],[2,1,1,2,2,2,1,1,1,2,2,2,2,2,1],[0,1,1,1,0,0,1,1,1,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    test_in = [[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],[2,0,2,2,2,2,0,2,0,2,2,0,2,2,2],[2,0,0,2,2,2,0,0,0,2,2,0,0,0,2],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,5,0,0,0,5,0,5,0],[0,5,0,0,0,0,0,5,5,0,0,5,5,5,0],[0,5,5,5,0,0,0,5,0,0,0,5,5,5,0],[0,5,5,5,5,0,0,5,5,0,0,5,5,5,0]]
    test_out = [[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],[2,1,2,2,2,2,1,2,1,2,2,1,2,2,2],[2,1,1,2,2,2,1,1,1,2,2,1,1,1,2],[0,1,0,0,0,0,1,1,1,0,0,1,1,1,1],[0,1,1,0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    ok = True
    for name, inp, expected in [("Ex0", ex0_in, ex0_out), ("Ex1", ex1_in, ex1_out), ("Test", test_in, test_out)]:
        result = transform(inp)
        if result == expected:
            print(f"{name}: PASS")
        else:
            ok = False
            print(f"{name}: FAIL")
            for r in range(len(expected)):
                if result[r] != expected[r]:
                    print(f"  Row {r}: got  {result[r]}")
                    print(f"         want {expected[r]}")

    print("\nSOLVED" if ok else "\nFAILED")
