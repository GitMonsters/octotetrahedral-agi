from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    colors = Counter(v for row in grid for v in row)
    bg = colors.most_common(1)[0][0]
    
    # Find wall color (2nd most common or the one forming full rows/cols)
    # Wall cols: columns that are entirely wall_color (non-bg, non-signal)
    wall_color = None
    signal_color = None
    for v, cnt in colors.most_common():
        if v == bg: continue
        if wall_color is None:
            # Try as wall: check if it forms full cols
            full_cols = [c for c in range(W) if all(grid[r][c] == v for r in range(H))]
            full_rows = [r for r in range(H) if all(grid[r][c] == v for c in range(W))]
            if full_cols or full_rows:
                wall_color = v
            else:
                signal_color = v
    
    if wall_color is None:
        return out
    
    # Find signal pixels
    signals = [(r, c) for r in range(H) for c in range(W) if grid[r][c] not in (bg, wall_color)]
    # Find wall columns and rows (needed for both signal and no-signal paths)
    wall_cols = sorted([c for c in range(W) if all(grid[r][c] == wall_color for r in range(H))])
    wall_rows = sorted([r for r in range(H) if all(grid[r][c] == wall_color for c in range(W))])
    if not signals:
        # Wall-only input: generate symmetric patterns based on wall positions
        if len(wall_rows) >= 2:
            w0 = wall_rows[0]
            w_gap = wall_rows[1] - wall_rows[0]
            g1_start = W // 2
            g2_start = W - w0 - 1
            for wi, wr in enumerate(wall_rows):
                g1_width = w_gap if wi == 0 else w0
                g2_width = w0
                groups = [(g1_start, g1_width), (g2_start, g2_width)]
                for gs, gw in groups:
                    # Adjacent rows get wall_color
                    for adj in [wr - 1, wr + 1]:
                        if 0 <= adj < H:
                            for c in range(gs, gs + gw):
                                out[adj][c] = wall_color
                    # Wall row gets holes (bg) at odd offsets within group
                    for offset in range(1, gw, 2):
                        out[wr][gs + offset] = bg
        return out
    signal_color = grid[signals[0][0]][signals[0][1]]
    
    for sr, sc in signals:
        sig_val = grid[sr][sc]
        
        # Find nearest wall col to the left
        left_walls = [c for c in wall_cols if c < sc]
        right_walls = [c for c in wall_cols if c > sc]
        top_walls = [r for r in wall_rows if r < sr]
        bot_walls = [r for r in wall_rows if r > sr]
        
        # Fire toward each adjacent wall
        walls_to_fire = []
        if left_walls:
            walls_to_fire.append(('left', left_walls[-1]))
        if right_walls:
            walls_to_fire.append(('right', right_walls[0]))
        if top_walls:
            walls_to_fire.append(('up', top_walls[-1]))
        if bot_walls:
            walls_to_fire.append(('down', bot_walls[0]))
        
        for direction, wall_pos in walls_to_fire:
            if direction == 'right':
                wc = wall_pos
                # Row sr: sc..wc-2 = sig, wc-1 = wall, wc = sig
                for c in range(sc, wc - 1):
                    out[sr][c] = sig_val
                if wc - 1 >= 0:
                    out[sr][wc - 1] = wall_color
                out[sr][wc] = sig_val
                # Adjacent rows
                for dr in [-1, 1]:
                    nr = sr + dr
                    if 0 <= nr < H and 0 <= wc - 1 < W:
                        out[nr][wc - 1] = wall_color
            
            elif direction == 'left':
                wc = wall_pos
                # Row sr: wc = sig, wc+1 = wall, wc+2..sc = sig
                out[sr][wc] = sig_val
                if wc + 1 < W:
                    out[sr][wc + 1] = wall_color
                for c in range(wc + 2, sc + 1):
                    out[sr][c] = sig_val
                # Adjacent rows
                for dr in [-1, 1]:
                    nr = sr + dr
                    if 0 <= nr < H and 0 <= wc + 1 < W:
                        out[nr][wc + 1] = wall_color
            
            elif direction == 'down':
                wr = wall_pos
                for r in range(sr, wr - 1):
                    out[r][sc] = sig_val
                if wr - 1 >= 0:
                    out[wr - 1][sc] = wall_color
                out[wr][sc] = sig_val
                for dc in [-1, 1]:
                    nc = sc + dc
                    if 0 <= nc < W and 0 <= wr - 1 < H:
                        out[wr - 1][nc] = wall_color
            
            elif direction == 'up':
                wr = wall_pos
                out[wr][sc] = sig_val
                if wr + 1 < H:
                    out[wr + 1][sc] = wall_color
                for r in range(wr + 2, sr + 1):
                    out[r][sc] = sig_val
                for dc in [-1, 1]:
                    nc = sc + dc
                    if 0 <= nc < W and 0 <= wr + 1 < H:
                        out[wr + 1][nc] = wall_color
    
    return out
