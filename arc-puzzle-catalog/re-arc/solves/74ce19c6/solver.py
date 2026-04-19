def transform(input_grid):
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color (most common)
    flat = [c for r in input_grid for c in r]
    bg = Counter(flat).most_common(1)[0][0]

    # Find non-background cells
    fg_cells = set(
        (r, c) for r in range(rows) for c in range(cols) if input_grid[r][c] != bg
    )

    if not fg_cells:
        # Uniform grid: if any dimension divisible by 3 -> 6, else bg color
        if rows % 3 == 0 or cols % 3 == 0:
            return [[6]]
        else:
            return [[bg]]

    # Find bounding box
    min_r = min(r for r, c in fg_cells)
    max_r = max(r for r, c in fg_cells)
    min_c = min(c for r, c in fg_cells)
    max_c = max(c for r, c in fg_cells)
    bbox_h = max_r - min_r + 1
    bbox_w = max_c - min_c + 1

    # Normalize shape
    shape = set((r - min_r, c - min_c) for r, c in fg_cells)

    # Try to decompose into a 3x3 block grid
    for bs in range(1, max(bbox_h, bbox_w) + 1):
        if bbox_h % bs != 0 or bbox_w % bs != 0:
            continue
        if bbox_h // bs != 3 or bbox_w // bs != 3:
            continue

        # Check if each block is uniformly filled or empty
        grid = [[0] * 3 for _ in range(3)]
        valid = True
        for br in range(3):
            for bc in range(3):
                filled = sum(
                    1
                    for dr in range(bs)
                    for dc in range(bs)
                    if (br * bs + dr, bc * bs + dc) in shape
                )
                if filled == 0:
                    grid[br][bc] = 0
                elif filled == bs * bs:
                    grid[br][bc] = 1
                else:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            plus = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
            x_pat = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]

            if grid == plus:
                return [[4]]
            elif grid == x_pat:
                return [[2]]
            else:
                return [[6]]

    # Fallback: not a recognized 3x3 pattern
    return [[6]]
