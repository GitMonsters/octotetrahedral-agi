"""
ARC Puzzle 007a5851 Solver

Pattern: Corner markers define diagonal zones with alternating stripe patterns.
From each corner, diagonal waves emanate creating checkerboard-like stripes
that meet at a diagonal line between the markers.
"""

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    flat = [c for row in grid for c in row]
    bg = max(set(flat), key=flat.count)
    
    # Find corner markers (non-background colors at corners/edges)
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                markers.append((r, c, grid[r][c]))
    
    # Create output grid
    output = [[bg for _ in range(cols)] for _ in range(rows)]
    
    if len(markers) == 2:
        # Two markers - diagonal dividing line
        (r1, c1, color1), (r2, c2, color2) = markers
        
        for r in range(rows):
            for c in range(cols):
                # Calculate diagonal distance from each marker
                # Using Manhattan-like diagonal distance
                d1 = max(abs(r - r1), abs(c - c1))
                d2 = max(abs(r - r2), abs(c - c2))
                
                # Determine which zone this cell belongs to
                if d1 < d2:
                    color = color1
                    dist = d1
                    # Direction from marker
                    dr = 1 if r2 > r1 else -1
                    dc = 1 if c2 > c1 else -1
                elif d2 < d1:
                    color = color2
                    dist = d2
                    dr = 1 if r1 > r2 else -1
                    dc = 1 if c1 > c2 else -1
                else:
                    # On the diagonal - belongs to neither exclusively
                    output[r][c] = bg
                    continue
                
                # Determine if this cell should be colored
                # Even rows from perspective: solid stripe
                # Odd rows from perspective: checkerboard
                row_from_marker = abs(r - (r1 if color == color1 else r2))
                col_from_marker = abs(c - (c1 if color == color1 else c2))
                
                if row_from_marker % 2 == 0:
                    # Even row - fill cells within distance
                    if col_from_marker <= row_from_marker:
                        output[r][c] = color
                else:
                    # Odd row - checkerboard pattern
                    if col_from_marker % 2 == 0 and col_from_marker < row_from_marker:
                        output[r][c] = color
                        
    elif len(markers) == 3:
        # Three markers - more complex pattern
        # Find which markers are paired (same color or opposite corners)
        m1, m2, m3 = markers
        
        for r in range(rows):
            for c in range(cols):
                # Check each marker's influence
                for mr, mc, mcolor in markers:
                    # Calculate row/col distance
                    row_dist = abs(r - mr)
                    col_dist = abs(c - mc)
                    
                    # Determine the "reach" direction of this marker
                    # Find if this cell is in the marker's zone
                    
                    # Check if cell is closer to this marker than others
                    my_diag = max(row_dist, col_dist)
                    others_closer = False
                    for mr2, mc2, mc2_color in markers:
                        if (mr2, mc2) != (mr, mc):
                            other_diag = max(abs(r - mr2), abs(c - mc2))
                            if other_diag < my_diag:
                                others_closer = True
                                break
                    
                    if not others_closer or my_diag == 0:
                        # This marker has influence here
                        if row_dist % 2 == 0:
                            if col_dist <= row_dist:
                                output[r][c] = mcolor
                        else:
                            if col_dist % 2 == 0 and col_dist < row_dist:
                                output[r][c] = mcolor
    
    return output


def transform_v2(grid):
    """Alternative approach: treat each marker as emitting diagonal waves"""
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color
    flat = [c for row in grid for c in row]
    bg = max(set(flat), key=flat.count)
    
    # Find markers
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                markers.append((r, c, grid[r][c]))
    
    output = [[bg for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            # For each marker, calculate if this cell should be colored
            for mr, mc, mcolor in markers:
                row_d = r - mr
                col_d = c - mc
                
                # Check if this cell is in the "forward" direction from marker
                # The pattern extends in a specific diagonal direction
                
                # Chebyshev distance
                cheb = max(abs(row_d), abs(col_d))
                
                # Check if closer to this marker than any other
                is_closest = True
                for mr2, mc2, _ in markers:
                    if (mr2, mc2) != (mr, mc):
                        cheb2 = max(abs(r - mr2), abs(c - mc2))
                        if cheb2 < cheb:
                            is_closest = False
                            break
                
                if is_closest:
                    row_dist = abs(row_d)
                    col_dist = abs(col_d)
                    
                    # Pattern: on even rows, fill up to row distance
                    # on odd rows, checkerboard
                    if row_dist % 2 == 0:
                        if col_dist <= row_dist:
                            output[r][c] = mcolor
                    else:
                        if col_dist % 2 == 0 and col_dist < row_dist:
                            output[r][c] = mcolor
    
    return output


def transform_v3(grid):
    """Refined approach based on careful analysis"""
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color
    flat = [c for row in grid for c in row]
    bg = max(set(flat), key=flat.count)
    
    # Find markers
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                markers.append((r, c, grid[r][c]))
    
    output = [[bg for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            best_marker = None
            best_dist = float('inf')
            
            for mr, mc, mcolor in markers:
                # Chebyshev distance
                dist = max(abs(r - mr), abs(c - mc))
                if dist < best_dist:
                    best_dist = dist
                    best_marker = (mr, mc, mcolor)
            
            if best_marker:
                mr, mc, mcolor = best_marker
                row_dist = abs(r - mr)
                col_dist = abs(c - mc)
                
                # Alternating pattern based on row distance
                if row_dist % 2 == 0:
                    # Even: fill from start up to diagonal
                    if col_dist <= row_dist:
                        output[r][c] = mcolor
                else:
                    # Odd: every other column
                    if col_dist % 2 == 0 and col_dist < row_dist:
                        output[r][c] = mcolor
    
    return output


# Test function
if __name__ == "__main__":
    import json
    
    task = json.load(open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json'))['007a5851']
    
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform_v3(inp)
        
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            # Show differences
            for r in range(min(5, len(expected))):
                print(f"  Row {r} expected: {expected[r][:15]}...")
                print(f"  Row {r} got:      {result[r][:15]}...")
