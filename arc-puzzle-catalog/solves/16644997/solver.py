def transform(grid):
    from collections import Counter
    N = len(grid)
    flat = [grid[r][c] for r in range(N) for c in range(N)]
    bg = Counter(flat).most_common(1)[0][0]

    corners = {}
    for name, (rr, cr) in [
        ('TL', (range(N//2), range(N//2))),
        ('TR', (range(N//2), range(N//2, N))),
        ('BL', (range(N//2, N), range(N//2))),
        ('BR', (range(N//2, N), range(N//2, N))),
    ]:
        corners[name] = sum(1 for r in rr for c in cr if grid[r][c] != bg)
    pattern_corner = max(corners, key=corners.get)

    non_bg = [(r, c) for r in range(N) for c in range(N) if grid[r][c] != bg]
    if pattern_corner == 'TL':
        ir, ic = 0, 0
        P_size = max(max(r for r,_ in non_bg) + 1, max(c for _,c in non_bg) + 1)
    elif pattern_corner == 'TR':
        ir, ic = 0, N - 1
        P_size = max(max(r for r,_ in non_bg) + 1, N - min(c for _,c in non_bg))
    elif pattern_corner == 'BL':
        ir, ic = N - 1, 0
        P_size = max(N - min(r for r,_ in non_bg), max(c for _,c in non_bg) + 1)
    else:
        ir, ic = N - 1, N - 1
        P_size = max(N - min(r for r,_ in non_bg), N - min(c for _,c in non_bg))

    shell_colors = {}
    for r, c in non_bg:
        d = max(abs(r - ir), abs(c - ic))
        if d not in shell_colors:
            shell_colors[d] = grid[r][c]

    out_N = 2 * N
    oir = 0 if ir == 0 else out_N - 1
    oic = 0 if ic == 0 else out_N - 1
    output = [[0]*out_N for _ in range(out_N)]
    for r in range(out_N):
        for c in range(out_N):
            d = max(abs(r - oir), abs(c - oic))
            output[r][c] = shell_colors.get(d % P_size, bg)
    return output
