from collections import Counter

def transform(input_grid):
    R = len(input_grid)
    C = len(input_grid[0])
    P = 2 * (R - 1)

    # Find background color (most common)
    color_count = Counter()
    for row in input_grid:
        for c in row:
            color_count[c] += 1
    bg_color = color_count.most_common(1)[0][0]

    # Find seed positions (non-background cells)
    seeds = []
    seed_color = None
    for r in range(R):
        for c in range(C):
            if input_grid[r][c] != bg_color:
                seeds.append((r, c))
                seed_color = input_grid[r][c]

    # Uniform grid: use all 4 corners as seeds
    if not seeds:
        seed_color = bg_color
        seeds = [(0, 0), (0, C - 1), (R - 1, 0), (R - 1, C - 1)]

    # Compute the set of diagonal residues mod P
    S = set()
    for (r0, c0) in seeds:
        S.add((r0 + c0) % P)
        S.add((c0 - r0) % P)

    # Generate output
    output = []
    for r in range(R):
        row = []
        for c in range(C):
            u = (r + c) % P
            v = (c - r) % P
            if u in S or v in S:
                row.append(seed_color)
            else:
                row.append(5)
        output.append(row)

    return output
