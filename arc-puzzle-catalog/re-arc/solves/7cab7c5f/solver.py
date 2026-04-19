from collections import Counter, deque

def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    def find_comps(color):
        visited = set(); comps = []
        for r in range(H):
            for c in range(W):
                if grid[r][c] == color and (r, c) not in visited:
                    q = deque([(r, c)]); visited.add((r, c)); comp = [(r, c)]
                    while q:
                        nr, nc = q.popleft()
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            rr, cc = nr+dr, nc+dc
                            if 0<=rr<H and 0<=cc<W and grid[rr][cc]==color and (rr,cc) not in visited:
                                visited.add((rr,cc)); q.append((rr,cc)); comp.append((rr,cc))
                    comps.append(comp)
        return comps

    # Find 8-pair
    comps8 = find_comps(8)
    eight_pair = next(c for c in comps8 if len(c) == 2)
    r8 = sorted(eight_pair)
    # Determine orientation
    rows8 = sorted(set(r for r, c in r8))
    cols8 = sorted(set(c for r, c in r8))
    eight_horiz = (len(rows8) == 1)  # horizontal = same row

    # Find 7-pairs: connected comps of 7 with size 2
    comps7 = find_comps(7)
    seven_pairs = [c for c in comps7 if len(c) == 2]

    # Filter by same orientation as 8-pair, pick furthest
    def pair_orientation(comp):
        rs = set(r for r, c in comp)
        return len(rs) == 1  # True=horiz, False=vert

    # Center of 8-pair
    c8r = sum(r for r, c in r8) / 2
    c8c = sum(c for r, c in r8) / 2

    candidates = [p for p in seven_pairs if pair_orientation(p) == eight_horiz]
    if not candidates:
        candidates = seven_pairs

    if not candidates:
        return [row[:] for row in grid]
    seven_pair = max(candidates, key=lambda p: abs(sum(r for r,c in p)/2 - c8r) + abs(sum(c for r,c in p)/2 - c8c))
    r7 = sorted(seven_pair)

    rows7 = sorted(set(r for r, c in r7))
    cols7 = sorted(set(c for r, c in r7))

    out = [row[:] for row in grid]

    if eight_horiz:
        # Both horizontal (same row)
        erow = rows8[0]
        if cols7 == cols8:
            # Same cols -> C-shape
            row1, row2 = (erow, rows7[0]) if erow < rows7[0] else (rows7[0], erow)
            c1, c2 = cols8[0], cols8[1]
            row_sep = row2 - row1
            # Determine opening direction
            if c2 >= W - 1 - c1:  # closer to right edge
                left_col = c2 - row_sep // 2
                # Top arm: row1 from left_col to c1-1
                for c in range(left_col, c1):
                    out[row1][c] = 8
                # Vertical: col left_col from row1+1 to row2-1
                for r in range(row1+1, row2):
                    out[r][left_col] = 8
                # Bottom arm: row2 from left_col to c1-1
                for c in range(left_col, c1):
                    out[row2][c] = 8
            else:
                right_col = c1 + row_sep // 2
                for c in range(c2+1, right_col+1):
                    out[row1][c] = 8
                for r in range(row1+1, row2):
                    out[r][right_col] = 8
                for c in range(c2+1, right_col+1):
                    out[row2][c] = 8
        else:
            # Different cols -> L-shape
            # 8-pair at row erow, cols cols8. 7-pair at row rows7[0], cols cols7.
            r_8, r_7 = erow, rows7[0]
            c8a, c8b = min(cols8), max(cols8)
            c7a, c7b = min(cols7), max(cols7)
            # L from right of 8-pair down to 7-pair's row, then across
            if r_8 < r_7:
                arm_col = c8b + 1
                # Vertical: arm_col from r_8 to r_7
                for r in range(r_8, r_7+1):
                    out[r][arm_col] = 8
                # Horizontal: r_7 from arm_col+1 to c7a-1
                for c in range(arm_col+1, c7a):
                    out[r_7][c] = 8
            else:
                arm_col = c8a - 1
                for r in range(r_7, r_8+1):
                    out[r][arm_col] = 8
                for c in range(c7b+1, arm_col):
                    out[r_7][c] = 8
    else:
        # Both vertical (same col)
        ecol = cols8[0]
        if rows7 == rows8:
            # Same rows -> U-shape
            r1, r2 = rows8[0], rows8[1]
            c1, c2 = (ecol, cols7[0]) if ecol < cols7[0] else (cols7[0], ecol)
            col_sep = c2 - c1
            # Direction: away from nearest edge
            dist_top = r1
            dist_bot = H - 1 - r2
            if dist_top <= dist_bot:  # opens downward
                bottom_row = r1 + col_sep - 1
                for r in range(r2+1, bottom_row+1):
                    out[r][c1] = 8
                    out[r][c2] = 8
                for c in range(c1+1, c2):
                    out[bottom_row][c] = 8
            else:  # opens upward
                top_row = r2 - col_sep + 1
                for r in range(top_row, r1):
                    out[r][c1] = 8
                    out[r][c2] = 8
                for c in range(c1+1, c2):
                    out[top_row][c] = 8
        else:
            # Different rows -> L-shape for vertical pairs
            c_8 = ecol
            c_7 = cols7[0]
            r8a, r8b = rows8[0], rows8[1]
            r7a, r7b = rows7[0], rows7[1]
            if r8a < r7a:
                arm_row = r8b + 1
                for r in range(r8b+1, r7a+1):
                    out[r][c_8] = 8
                for r in range(arm_row, r7b):
                    pass
                for c in range(min(c_8,c_7)+1, max(c_8,c_7)):
                    out[r7a][c] = 8

    return out
