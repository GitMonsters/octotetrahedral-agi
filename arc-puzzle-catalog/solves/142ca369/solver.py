"""Solver for 142ca369 — billiard ball diagonal rays from L-shapes bouncing off lines/pixels"""
import json
import numpy as np
from typing import List, Tuple, Set, Dict

def solve(grid: List[List[int]]) -> List[List[int]]:
    g = np.array(grid)
    H, W = g.shape
    result = g.copy()
    
    # Find connected components of same-color non-zero cells
    visited = set()
    shapes = []
    
    for r in range(H):
        for c in range(W):
            if g[r][c] != 0 and (r,c) not in visited:
                color = g[r][c]
                # BFS to find connected component
                comp = []
                queue = [(r,c)]
                visited.add((r,c))
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr,cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<H and 0<=nc<W and (nr,nc) not in visited and g[nr][nc]==color:
                            visited.add((nr,nc))
                            queue.append((nr,nc))
                shapes.append({'cells': comp, 'color': color})
    
    # Classify shapes
    for s in shapes:
        cells = s['cells']
        n = len(cells)
        rows = [r for r,c in cells]
        cols = [c for r,c in cells]
        
        if n == 1:
            s['type'] = 'pixel'
            s['pos'] = cells[0]
        elif n == 3:
            rspan = max(rows) - min(rows)
            cspan = max(cols) - min(cols)
            if rspan == 1 and cspan == 1:
                s['type'] = 'L'
                # Find corner (cell adjacent to both others)
                for cell in cells:
                    others = [c2 for c2 in cells if c2 != cell]
                    adj = sum(1 for o in others if abs(o[0]-cell[0])+abs(o[1]-cell[1])==1)
                    if adj == 2:
                        s['corner'] = cell
                        arm1 = (others[0][0]-cell[0], others[0][1]-cell[1])
                        arm2 = (others[1][0]-cell[0], others[1][1]-cell[1])
                        s['away_dir'] = (-(arm1[0]+arm2[0]), -(arm1[1]+arm2[1]))
                        break
            elif rspan == 0:
                s['type'] = 'hline'
                s['row'] = rows[0]
                s['cols'] = sorted(cols)
                s['mid'] = (rows[0], sorted(cols)[1])
            elif cspan == 0:
                s['type'] = 'vline'
                s['col'] = cols[0]
                s['rows'] = sorted(rows)
                s['mid'] = (sorted(rows)[1], cols[0])
        else:
            s['type'] = 'other'
    
    # Group shapes by color
    color_groups = {}
    for s in shapes:
        color = s['color']
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(s)
    
    # Identify L-shapes as starting points
    l_shapes = [s for s in shapes if s['type'] == 'L']
    lines = [s for s in shapes if s['type'] in ('hline', 'vline')]
    pixels = [s for s in shapes if s['type'] == 'pixel']
    
    # Build occupied set (all non-zero cells in input)
    occupied = set()
    for r in range(H):
        for c in range(W):
            if g[r][c] != 0:
                occupied.add((r,c))
    
    # Fire billiard balls from L-shapes
    for l in l_shapes:
        fire_billiard(result, l['corner'], l['away_dir'], l['color'], 
                      lines, pixels, occupied, H, W)
    
    # Handle line shapes that weren't hit by billiard balls
    # (they fire their own diagonals independently)
    for line in lines:
        if line.get('hit'):
            continue
        # This line wasn't hit by any billiard ball
        # Determine extension direction and fire
        # For paired lines (same color has 2 shapes), extend toward each other
        color = line['color']
        partners = [s for s in color_groups[color] if s != line and s['type'] in ('hline', 'vline')]
        
        if partners:
            partner = partners[0]
            # Extension toward partner
            if line['type'] == 'vline':
                mid_r, mid_c = line['mid']
                partner_c = partner.get('col', partner.get('cols', [0])[1] if partner['type']=='hline' else 0)
                if partner['type'] == 'vline':
                    partner_c = partner['col']
                elif partner['type'] == 'hline':
                    partner_c = partner['cols'][1]
                
                # Determine row direction: away from partner
                partner_r = partner.get('row', partner.get('rows', [0])[1] if partner['type']=='vline' else 0)
                if partner['type'] == 'vline':
                    partner_r = partner['rows'][1]
                elif partner['type'] == 'hline':
                    partner_r = partner['row']
                
                # Extension direction: toward grid center for column
                ext_dc = 1 if mid_c < W/2 else -1
                ext_r, ext_c = mid_r, mid_c + ext_dc
                
                # Diagonal direction
                diag_dr = 1 if mid_r < partner_r else -1
                # Actually: away from partner
                diag_dr = -1 if mid_r < partner_r else 1
                diag_dc = ext_dc
                
            elif line['type'] == 'hline':
                mid_r, mid_c = line['mid']
                partner_r = partner.get('row', partner.get('rows', [0])[1]) if partner['type']=='hline' else partner.get('rows', [0])[1]
                if partner['type'] == 'hline':
                    partner_r = partner['row']
                elif partner['type'] == 'vline':
                    partner_r = partner['rows'][1]
                
                ext_dr = 1 if mid_r < H/2 else -1
                ext_r, ext_c = mid_r + ext_dr, mid_c
                
                diag_dc = 1 if mid_c < partner_r else -1  # bug: comparing col to row
                # Away from partner in column direction... 
                # Actually determine diagonal: perpendicular to line + extension direction
                partner_c = partner.get('col', partner.get('cols', [0])[1])
                if partner['type'] == 'hline':
                    partner_c = partner['cols'][1]
                elif partner['type'] == 'vline':
                    partner_c = partner['col']
                    
                diag_dr = ext_dr
                diag_dc = -1 if mid_c < partner_c else 1
                # Away from partner
                diag_dc = -1 if mid_c > partner_c else 1
                # Hmm... this is getting messy
            
            # Fire the diagonal
            ext_pos = (ext_r, ext_c)
            if (ext_pos not in occupied and 0 <= ext_r < H and 0 <= ext_c < W):
                result[ext_r][ext_c] = line['color']
                occupied.add(ext_pos)
                # Draw diagonal
                cr, cc = ext_r + diag_dr, ext_c + diag_dc
                while 0 <= cr < H and 0 <= cc < W and (cr,cc) not in occupied:
                    result[cr][cc] = line['color']
                    occupied.add((cr,cc))
                    cr += diag_dr
                    cc += diag_dc
    
    return result.tolist()

def fire_billiard(result, start, direction, color, lines, pixels, occupied, H, W):
    """Fire a billiard ball from start in direction, bouncing off lines and pixels."""
    r, c = start
    dr, dc = direction
    
    # Build lookup for lines and pixels
    line_cells = {}
    for line in lines:
        for cell in line['cells']:
            line_cells[cell] = line
    
    pixel_map = {}
    for p in pixels:
        pixel_map[p['pos']] = p
    
    max_steps = H * W * 4  # prevent infinite loops
    steps = 0
    
    while steps < max_steps:
        nr, nc = r + dr, c + dc
        steps += 1
        
        # Check if out of bounds
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        
        # Check if hitting an occupied cell
        if (nr, nc) in occupied:
            # Check if it's a line we can bounce off
            if (nr, nc) in line_cells:
                line = line_cells[(nr, nc)]
                if not line.get('hit'):
                    line['hit'] = True
                    # Bounce: reverse appropriate component
                    if line['type'] == 'vline':
                        dc = -dc  # reverse column direction
                    elif line['type'] == 'hline':
                        dr = -dr  # reverse row direction
                    color = line['color']
                    # The current cell (r, c) becomes the extension
                    # Continue from (r, c) in new direction
                    # Actually, we need to place the extension and continue
                    # The extension is at (nr-dr_bounce, nc-dc_bounce) = current (r,c)?
                    # No: we stay at (r,c) which is already drawn, and continue from (r,c) in new direction
                    # Actually (r,c) IS the extension point - it gets the new color
                    if result[r][c] == 0 or (r,c) == start:
                        result[r][c] = color  # Recolor extension point with line's color
                    # Actually the extension IS the last cell we drew
                    # Let me reconsider: the ball is at (r,c), next step hits (nr,nc) which is a line
                    # The ball doesn't move to (nr,nc), it bounces
                    # The cell (r,c) was already colored by previous step
                    # We need to recolor it? Or leave it?
                    # Actually in Train 2, (6,6) gets color 1 (blue), not 4 (yellow)
                    # So the extension cell (where the ball is when it bounces) gets the LINE's color
                    if (r,c) != start:
                        result[r][c] = color
                        # Remove old color from occupied set (it was the ball's previous color)
                    # Continue from (r,c) in new direction
                    continue
                else:
                    break  # Already hit this line, stop
            
            # Check if hitting a pixel
            # Actually for pixels in Train 0, the ball doesn't directly hit the pixel
            # It reflects when reaching the pixel's column
            # Let me handle this differently
            
            # For now, just stop at occupied cells
            break
        
        # Check for pixel reflection (single pixel)
        # The ball reflects when it reaches the same row or column as a pixel of the same color
        reflected = False
        for p in pixels:
            if p['color'] == color:
                pr, pc = p['pos']
                if nc == pc and nr != pr:
                    # Reached pixel's column, reflect row direction
                    dr = -dr
                    reflected = True
                    break
                if nr == pr and nc != pc:
                    # Reached pixel's row, reflect column direction
                    dc = -dc
                    reflected = True
                    break
        
        # Place the ball
        result[nr][nc] = color
        occupied.add((nr, nc))
        r, c = nr, nc
    
    return

# Test
path = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/142ca369.json"
with open(path) as f:
    task = json.load(f)

for i, ex in enumerate(task['train']):
    out = solve(ex['input'])
    expected = ex['output']
    match = out == expected
    print(f"Train {i}: {'PASS ✓' if match else 'FAIL'}")
    if not match:
        ea = np.array(expected)
        ga = np.array(out)
        diff = ea != ga
        ndiff = np.sum(diff)
        print(f"  Diffs: {ndiff} cells")
        if ndiff < 20:
            for r in range(diff.shape[0]):
                for c in range(diff.shape[1]):
                    if diff[r][c]:
                        print(f"    ({r},{c}): expected={ea[r][c]}, got={ga[r][c]}")
