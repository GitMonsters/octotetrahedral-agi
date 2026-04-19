from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find connected components (4-connectivity)
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                pixels = []
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if 0 <= cr < rows and 0 <= cc < cols and not visited[cr][cc] and grid[cr][cc] == color:
                        visited[cr][cc] = True
                        pixels.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((cr + dr, cc + dc))
                components.append((color, pixels))

    # Main shape = largest connected component
    main_comp = max(components, key=lambda x: len(x[1]))
    main_color, main_pixels = main_comp

    # Bounding box of main shape
    min_r = min(r for r, c in main_pixels)
    max_r = max(r for r, c in main_pixels)
    min_c = min(c for r, c in main_pixels)
    max_c = max(c for r, c in main_pixels)
    H = max_r - min_r + 1
    W = max_c - min_c + 1

    # Shape pattern as relative positions within bounding box
    shape = set()
    for r, c in main_pixels:
        shape.add((r - min_r, c - min_c))

    stride_r = H + 1
    stride_c = W + 1

    # Each marker pixel defines a tiling direction and color
    directions = {}
    for comp in components:
        if comp is main_comp:
            continue
        comp_color, comp_pixels = comp
        for mr, mc in comp_pixels:
            dr = mr - min_r
            dc = mc - min_c
            pr = dr % stride_r
            pc = dc % stride_c
            if (pr, pc) in shape:
                i = (dr - pr) // stride_r
                j = (dc - pc) // stride_c
                dir_r = 0 if i == 0 else (1 if i > 0 else -1)
                dir_c = 0 if j == 0 else (1 if j > 0 else -1)
                if (dir_r, dir_c) != (0, 0):
                    directions[(dir_r, dir_c)] = comp_color

    # Build output: keep main shape, clear markers
    output = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if output[r][c] != bg and output[r][c] != main_color:
                output[r][c] = bg

    # Tile copies of the main shape in each direction with the marker color
    for (dr, dc), color in directions.items():
        k = 1
        while True:
            tr = min_r + dr * k * stride_r
            tc = min_c + dc * k * stride_c

            any_in_bounds = False
            for pr, pc in shape:
                if 0 <= tr + pr < rows and 0 <= tc + pc < cols:
                    any_in_bounds = True
                    break

            if not any_in_bounds:
                break

            for pr, pc in shape:
                rr = tr + pr
                cc = tc + pc
                if 0 <= rr < rows and 0 <= cc < cols:
                    output[rr][cc] = color

            k += 1

    return output
