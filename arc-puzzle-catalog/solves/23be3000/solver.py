from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    freq = Counter(grid[r][c] for r in range(H) for c in range(W))
    colors = [c for c, _ in freq.most_common()]
    region_a = colors[0]
    region_b = colors[1]
    blob_color = colors[2] if len(colors) > 2 else None

    output = [row[:] for row in grid]

    def row_dominant(r):
        return Counter(grid[r][c] for c in range(W)).most_common(1)[0][0]

    def col_dominant(c):
        return Counter(grid[r][c] for r in range(H)).most_common(1)[0][0]

    boundary_row = None
    for r in range(H - 1):
        if row_dominant(r) != row_dominant(r + 1):
            boundary_row = r
            break

    boundary_col = None
    for c in range(W - 1):
        if col_dominant(c) != col_dominant(c + 1):
            boundary_col = c
            break

    if boundary_row is not None:
        top_color = row_dominant(0)
        bot_color = row_dominant(H - 1)

        intrusion = [(r, c) for r in range(boundary_row + 1)
                     for c in range(W) if grid[r][c] == bot_color]
        if not intrusion:
            intrusion = [(r, c) for r in range(boundary_row + 1, H)
                         for c in range(W) if grid[r][c] == top_color]

        blob = [(r, c) for r in range(H) for c in range(W)
                if grid[r][c] == blob_color]

        if intrusion and blob:
            blob_min_r = min(r for r, c in blob)
            blob_min_c = min(c for r, c in blob)
            blob_shape = [(r - blob_min_r, c - blob_min_c) for r, c in blob]
            intr_min_r = min(r for r, c in intrusion)
            intr_min_c = min(c for r, c in intrusion)

            for r, c in blob:
                output[r][c] = top_color if r <= boundary_row else bot_color
            for r, c in intrusion:
                output[r][c] = top_color if r <= boundary_row else bot_color

            # Always paint new marks as 7
            for dr, dc in blob_shape:
                nr, nc = intr_min_r + dr, intr_min_c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr][nc] = 7

    elif boundary_col is not None:
        left_color = col_dominant(0)
        right_color = col_dominant(W - 1)

        intrusion = [(r, c) for r in range(H) for c in range(boundary_col + 1)
                     if grid[r][c] == right_color]
        if not intrusion:
            intrusion = [(r, c) for r in range(H)
                         for c in range(boundary_col + 1, W)
                         if grid[r][c] == left_color]

        blob = [(r, c) for r in range(H) for c in range(W)
                if grid[r][c] == blob_color]

        if intrusion and blob:
            blob_min_r = min(r for r, c in blob)
            blob_min_c = min(c for r, c in blob)
            blob_shape = [(r - blob_min_r, c - blob_min_c) for r, c in blob]
            intr_min_r = min(r for r, c in intrusion)
            intr_min_c = min(c for r, c in intrusion)

            for r, c in blob:
                output[r][c] = left_color if c <= boundary_col else right_color
            for r, c in intrusion:
                output[r][c] = left_color if c <= boundary_col else right_color

            for dr, dc in blob_shape:
                nr, nc = intr_min_r + dr, intr_min_c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr][nc] = 7

    return output
