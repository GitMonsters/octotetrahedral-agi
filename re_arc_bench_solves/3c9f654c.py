"""
ARC Puzzle 3c9f654c Solver

Pattern: Each small cluster has a main shape and single-pixel markers.
The single pixels indicate expansion direction - the shape reflects/expands
across those marker pixels.
"""

def transform(grid):
    import copy
    from collections import deque
    
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)
    
    # Find background (most common color)
    color_count = {}
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] = color_count.get(grid[r][c], 0) + 1
    bg = max(color_count, key=color_count.get)
    
    # Find all non-background cells
    non_bg = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                non_bg.add((r, c))
    
    # Find connected clusters (8-connected)
    visited = set()
    clusters = []
    
    def get_cluster(start):
        cluster = set()
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            if (r, c) in cluster:
                continue
            cluster.add((r, c))
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if (nr, nc) in non_bg and (nr, nc) not in cluster:
                            queue.append((nr, nc))
        return cluster
    
    for cell in non_bg:
        if cell not in visited:
            cluster = get_cluster(cell)
            visited.update(cluster)
            clusters.append(cluster)
    
    # Process each cluster
    for cluster in clusters:
        # Group cells by color
        color_cells = {}
        for r, c in cluster:
            color = grid[r][c]
            if color not in color_cells:
                color_cells[color] = []
            color_cells[color].append((r, c))
        
        # Find main shape (largest connected group of same color)
        main_color = None
        main_cells = []
        for color, cells in color_cells.items():
            if len(cells) > len(main_cells):
                main_cells = cells
                main_color = color
        
        # Find marker cells (single isolated pixels of other colors)
        markers = []
        for color, cells in color_cells.items():
            if color != main_color:
                markers.extend([(r, c, color) for r, c in cells])
        
        if not main_cells or not markers:
            continue
        
        # Get bounding box of main shape
        min_r = min(r for r, c in main_cells)
        max_r = max(r for r, c in main_cells)
        min_c = min(c for r, c in main_cells)
        max_c = max(c for r, c in main_cells)
        
        # For each marker, reflect the main shape and the marker
        for mr, mc, marker_color in markers:
            # Determine reflection direction based on marker position relative to main shape
            
            # Reflect horizontally (across vertical axis through marker)
            if mc <= min_c or mc >= max_c:
                # Vertical reflection
                for r, c in main_cells:
                    new_c = 2 * mc - c
                    if 0 <= new_c < cols and 0 <= r < rows:
                        output[r][new_c] = main_color
                # Also reflect the marker
                for mr2, mc2, mc_color in markers:
                    if mc_color != marker_color:
                        new_c = 2 * mc - mc2
                        if 0 <= new_c < cols and 0 <= mr2 < rows:
                            output[mr2][new_c] = mc_color
            
            # Reflect vertically (across horizontal axis through marker)
            if mr <= min_r or mr >= max_r:
                # Horizontal reflection
                for r, c in main_cells:
                    new_r = 2 * mr - r
                    if 0 <= new_r < rows and 0 <= c < cols:
                        output[new_r][c] = main_color
                # Also reflect the marker
                for mr2, mc2, mc_color in markers:
                    if mc_color != marker_color:
                        new_r = 2 * mr - mr2
                        if 0 <= new_r < rows and 0 <= mc2 < cols:
                            output[new_r][mc2] = mc_color
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['3c9f654c']
    
    print("Testing on training examples:")
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        print(f"\nExample {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            print("Expected:")
            for row in expected:
                print(row)
            print("Got:")
            for row in result:
                print(row)
