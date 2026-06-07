def solve(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Edge-1 markers define lines: top/bottom edge → vertical; left/right edge → horizontal
    h_lines, v_lines = set(), set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 1:
                if r == 0 or r == H - 1:
                    v_lines.add(c)
                if c == 0 or c == W - 1:
                    h_lines.add(r)

    # Draw lines through zeros
    for r in range(H):
        for c in range(W):
            if (r in h_lines or c in v_lines) and result[r][c] == 0:
                result[r][c] = 1

    # Find blobs of 2; replace entire blob if intersected by a line
    vis: set = set()
    for sr in range(H):
        for sc in range(W):
            if grid[sr][sc] == 2 and (sr, sc) not in vis:
                blob = []
                stack = [(sr, sc)]
                while stack:
                    rr, cc = stack.pop()
                    if (rr, cc) in vis or not (0 <= rr < H and 0 <= cc < W) or grid[rr][cc] != 2:
                        continue
                    vis.add((rr, cc))
                    blob.append((rr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((rr + dr, cc + dc))
                hits = any(r in h_lines or c in v_lines for r, c in blob)
                if not hits:
                    # Also convert if blob is directly adjacent to any line cell
                    blob_set = set(blob)
                    for r, c in blob:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if (nr, nc) not in blob_set and (nr in h_lines or nc in v_lines):
                                hits = True
                                break
                        if hits:
                            break
                if hits:
                    for r, c in blob:
                        result[r][c] = 1

    return result
