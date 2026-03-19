"""
ARC Puzzle 24851717 Solver

Pattern: A diagonal color forms a zigzag "spine" through the grid via
connected anti-diagonals (r+c=k) and main-diagonals (r-c=k).
The triangular corner regions at each turn get filled with color 2.
"""

def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])
    
    out = [row[:] for row in grid]
    
    # Find the diagonal color - must form complete diagonals
    diag_colors = set()
    
    # Find all complete anti-diagonals (r+c=k)
    all_anti = {}  # color -> list of (k, cells)
    for k in range(H + W - 1):
        cells = [(r, k - r) for r in range(H) if 0 <= k - r < W]
        if len(cells) == W:
            vals = [grid[r][c] for r, c in cells]
            if len(set(vals)) == 1 and vals[0] != 2:
                color = vals[0]
                all_anti.setdefault(color, []).append((k, cells))
                diag_colors.add(color)
    
    # Find all complete main-diagonals (r-c=k)
    all_main = {}  # color -> list of (k, cells)
    for k in range(-(W - 1), H):
        cells = [(r, r - k) for r in range(H) if 0 <= r - k < W]
        if len(cells) == W:
            vals = [grid[r][c] for r, c in cells]
            if len(set(vals)) == 1 and vals[0] != 2:
                color = vals[0]
                all_main.setdefault(color, []).append((k, cells))
                diag_colors.add(color)
    
    # Process each diagonal color separately
    for diag_color in diag_colors:
        anti_diags = all_anti.get(diag_color, [])
        main_diags = all_main.get(diag_color, [])
        
        if not anti_diags and not main_diags:
            continue
        
        # Build the zigzag chain by sorting diagonals by their starting row
        # and tracking connections at vertices
        
        # Create vertex map: (r,c) -> list of (type, k, direction)
        # direction: 'in' if diagonal ends here, 'out' if diagonal starts here
        vertices = {}
        
        for k, cells in anti_diags:
            # Anti-diagonal goes from (r_low, c_high) to (r_high, c_low)
            start = cells[0]   # top-right
            end = cells[-1]    # bottom-left
            vertices.setdefault(start, []).append(('anti', k, 'out'))
            vertices.setdefault(end, []).append(('anti', k, 'in'))
        
        for k, cells in main_diags:
            # Main-diagonal goes from (r_low, c_low) to (r_high, c_high)
            start = cells[0]   # top-left
            end = cells[-1]    # bottom-right
            vertices.setdefault(start, []).append(('main', k, 'out'))
            vertices.setdefault(end, []).append(('main', k, 'in'))
        
        # Build ordered chain of diagonals
        diag_chain = []  # list of (type, k, cells)
        all_diags_sorted = []
        for k, cells in anti_diags:
            all_diags_sorted.append((cells[0][0], 'anti', k, cells))
        for k, cells in main_diags:
            all_diags_sorted.append((cells[0][0], 'main', k, cells))
        all_diags_sorted.sort()
        
        for _, dtype, k, cells in all_diags_sorted:
            diag_chain.append((dtype, k, cells))
        
        # Determine fill for each diagonal based on its position in the chain
        # Rule: Fill happens at corners (where direction changes)
        # The first diagonal's first part fills LEFT
        # After that, alternate based on turns
        
        for i, (dtype, k, cells) in enumerate(diag_chain):
            # Check connections to determine fill direction
            if dtype == 'anti':
                start_r, start_c = cells[0]   # top cell (col W-1)
                end_r, end_c = cells[-1]      # bottom cell (col 0)
                
                # Check if there's a main diagonal ending at start (coming from above-left)
                has_main_above = any(
                    mk == start_r - (W - 1) for mk, _ in main_diags
                )
                # Check if there's a main diagonal starting at end (going to below-right)
                has_main_below = any(
                    mk == end_r for mk, _ in main_diags
                )
                
                if not has_main_above and has_main_below:
                    # First anti-diagonal in a zigzag section - fill LEFT
                    for r, c in cells:
                        if r != end_r:  # Don't fill the shared vertex row
                            for col in range(c):
                                out[r][col] = 2
                elif has_main_above and has_main_below:
                    # Middle anti-diagonal - fill RIGHT (continuing from main)
                    for r, c in cells:
                        if r != start_r:  # Don't fill the shared vertex row
                            for col in range(c + 1, W):
                                out[r][col] = 2
                elif has_main_above and not has_main_below:
                    # Last anti-diagonal - fill LEFT (returning)
                    for r, c in cells:
                        if r != start_r:  # Don't fill the shared vertex row
                            for col in range(c):
                                out[r][col] = 2
            
            elif dtype == 'main':
                start_r, start_c = cells[0]   # top cell (col 0)
                end_r, end_c = cells[-1]      # bottom cell (col W-1)
                
                # Check if there's an anti-diagonal ending at start (coming from above-right)
                has_anti_above = any(
                    ak == start_r for ak, _ in anti_diags
                )
                # Check if there's an anti-diagonal starting at end (going to below-left)
                has_anti_below = any(
                    ak == end_r + W - 1 for ak, _ in anti_diags
                )
                
                if has_anti_above and has_anti_below:
                    # This main diagonal continues a fill to the RIGHT
                    for r, c in cells:
                        if r != start_r and r != end_r:  # Don't fill shared vertex rows
                            for col in range(c + 1, W):
                                out[r][col] = 2
                elif has_anti_above and not has_anti_below:
                    # This main diagonal continues a fill to the LEFT
                    # This happens when going "back up" in the zigzag
                    for r, c in cells:
                        if r != start_r:
                            for col in range(c):
                                out[r][col] = 2
        
        # Handle continuation rows after the last diagonal
        if anti_diags:
            last_anti_k = max(ak for ak, _ in anti_diags)
            last_anti_cells = [cells for ak, cells in anti_diags if ak == last_anti_k][0]
            last_anti_end_r = last_anti_cells[-1][0]
            
            # Check if this anti-diagonal has a main below it
            has_main_below = any(mk == last_anti_end_r for mk, _ in main_diags)
            
            if not has_main_below:
                # No main below - check for continuation rows
                for r in range(last_anti_end_r + 1, H):
                    for c in range(W):
                        if grid[r][c] == diag_color:
                            # Fill to the left of the continuation
                            for col in range(c):
                                out[r][col] = 2
                            break
    
    return out


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['24851717']
    
    all_pass = True
    for i, ex in enumerate(task['train']):
        inp, expected = ex['input'], ex['output']
        result = transform(inp)
        passed = result == expected
        all_pass = all_pass and passed
        print(f"Example {i}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print("  Expected:")
            for row in expected:
                print(f"    {row}")
            print("  Got:")
            for row in result:
                print(f"    {row}")
    
    print(f"\nAll tests: {'PASS' if all_pass else 'FAIL'}")
