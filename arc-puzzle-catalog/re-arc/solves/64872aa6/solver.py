def transform(input_grid):
    H, W = len(input_grid), len(input_grid[0])
    max_d = min(H, W) // 2

    # Get color at each depth (concentric rectangular rings)
    depth_colors = [input_grid[d][d] for d in range(max_d)]

    # Build ring structure: group consecutive same-color depths
    rings = []
    i = 0
    while i < max_d:
        color = depth_colors[i]
        thick = 1
        while i + thick < max_d and depth_colors[i + thick] == color:
            thick += 1
        rings.append((color, thick))
        i += thick

    n = len(rings)
    colors = [c for c, t in rings]
    thicknesses = [t for c, t in rings]

    # Right-rotate ring colors: innermost color becomes outermost
    rotated = [colors[-1]] + colors[:-1]

    # Build new ring list
    new_rings = [(rotated[j], thicknesses[j]) for j in range(n)]

    # Handle edge case: when outermost and innermost input colors are the same,
    # the rotation would leave the outermost ring unchanged and merge with the
    # second ring. Fix: peel off the outermost depth and assign it the original
    # second ring's color.
    if n >= 2 and rotated[0] == rotated[1]:
        border_color = colors[1]
        new_rings = [(border_color, 1)]
        if thicknesses[0] > 1:
            new_rings.append((rotated[0], thicknesses[0] - 1))
        for j in range(1, n):
            new_rings.append((rotated[j], thicknesses[j]))

    # Merge adjacent same-colored rings
    merged = [new_rings[0]]
    for j in range(1, len(new_rings)):
        if new_rings[j][0] == merged[-1][0]:
            merged[-1] = (merged[-1][0], merged[-1][1] + new_rings[j][1])
        else:
            merged.append(new_rings[j])

    # Build output per-depth color map
    output_depths = []
    for color, thick in merged:
        output_depths.extend([color] * thick)

    # Construct output grid
    output = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            d = min(r, c, H - 1 - r, W - 1 - c)
            output[r][c] = output_depths[d]

    return output
