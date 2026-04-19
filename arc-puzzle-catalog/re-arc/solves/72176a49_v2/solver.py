def transform(input_grid):
    """
    ARC Puzzle 72176a49 Solution
    
    Rules discovered:
    1. For each 9: place an 8 directly below it (if background)
    2. For each 8: place a 9 directly above it (if background)
    3. For each color with specific pattern of neighbors: 
       replicate that pattern wherever the neighbor configuration appears
    """
    from collections import Counter
    
    # Create mutable copy
    grid = [row[:] for row in input_grid]
    height, width = len(grid), len(grid[0])
    
    # Find background color (most common)
    flat = [cell for row in grid for cell in row]
    background = Counter(flat).most_common(1)[0][0]
    
    # Rule 0: L-shaped patterns (3 zeros + 1 non-background)
    # Find templates where non-background color at position X has 3 adjacent 0s
    # Then replicate that color wherever background + 3 zeros appears
    l_templates = {}  # Maps (position_index) -> color_value
    
    for r in range(height - 1):
        for c in range(width - 1):
            block = [grid[r][c], grid[r][c+1], grid[r+1][c], grid[r+1][c+1]]
            if block.count(0) == 3:
                # Find the non-zero value and its position
                positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
                for idx, val in enumerate(block):
                    if val != 0:
                        rel_pos = positions[idx]
                        if val != background:
                            # This is a template
                            l_templates[rel_pos] = val
    
    # Apply L-templates: find background + 3 zeros and place the template color
    for r in range(height - 1):
        for c in range(width - 1):
            block = [grid[r][c], grid[r][c+1], grid[r+1][c], grid[r+1][c+1]]
            if block.count(0) == 3:
                positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
                for idx, val in enumerate(block):
                    if val == background:
                        rel_pos = positions[idx]
                        if rel_pos in l_templates:
                            # Place the template color
                            dr, dc = rel_pos
                            grid[r + dr][c + dc] = l_templates[rel_pos]
    
    # Rule 1 & 2: Adjacent color pairs
    # Find pairs of colors that appear adjacent and replicate that adjacency
    # For each color, check if it has consistent adjacent colors in a specific direction
    
    # Collect color adjacencies
    from collections import defaultdict
    adjacencies = defaultdict(lambda: defaultdict(list))  # color -> direction -> [adjacent_color]
    
    for r in range(height):
        for c in range(width):
            if grid[r][c] != background:
                # Check 8 directions
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                            (-1, -1), (-1, 1), (1, -1), (1, 1)]
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        neighbor = grid[nr][nc]
                        if neighbor != background and neighbor != grid[r][c]:
                            adjacencies[grid[r][c]][(dr, dc)].append(neighbor)
    
    # Find consistent adjacency patterns (same color appears in same direction multiple times)
    consistent_pairs = []
    for color1, dir_map in adjacencies.items():
        for direction, neighbors in dir_map.items():
            if len(neighbors) >= 2:  # Multiple instances
                # Check if they're all the same color
                color_counts = Counter(neighbors)
                most_common_color, count = color_counts.most_common(1)[0]
                if count >= 2:
                    consistent_pairs.append((color1, direction, most_common_color))
    
    # Apply adjacency rules
    for color1, (dr, dc), color2 in consistent_pairs:
        for r in range(height):
            for c in range(width):
                if grid[r][c] == color1:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if grid[nr][nc] == background:
                            grid[nr][nc] = color2
    
    # Rule 3: Pattern replication (REVERSED)
    # If color A has pattern of color B at specific offsets,
    # then wherever we see color A, replicate color B at those offsets
    
    # Find all colors and their positions
    from collections import defaultdict
    color_positions = defaultdict(list)
    for r in range(height):
        for c in range(width):
            if grid[r][c] != background:
                color_positions[grid[r][c]].append((r, c))
    
    # For each color, find patterns of adjacent colors
    templates = []
    for center_color, positions in color_positions.items():
        for r, c in positions:
            # Check 3x3 neighborhood for other colors
            neighbor_patterns = defaultdict(list)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        neighbor_color = grid[nr][nc]
                        if neighbor_color != background and neighbor_color != center_color:
                            neighbor_patterns[neighbor_color].append((dr, dc))
            
            # Record templates with 2+ neighbor cells of same color
            for neighbor_color, offsets in neighbor_patterns.items():
                if len(offsets) >= 2:
                    templates.append({
                        'center_color': center_color,
                        'pattern_color': neighbor_color,
                        'offsets': sorted(set(offsets))
                    })
    
    # Deduplicate templates (keep the one with most offsets for each center-pattern pair)
    unique_templates = {}
    for t in templates:
        key = (t['center_color'], t['pattern_color'])
        if key not in unique_templates or len(t['offsets']) > len(unique_templates[key]['offsets']):
            unique_templates[key] = t
    
    # Apply templates: find center color and replicate pattern at offsets
    for template in unique_templates.values():
        center_color = template['center_color']
        pattern_color = template['pattern_color']
        offsets = template['offsets']
        
        # Find all instances of center_color and place pattern_color at offsets
        for r in range(height):
            for c in range(width):
                if grid[r][c] == center_color:
                    # Place pattern_color at all offsets (if background)
                    for dr, dc in offsets:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if grid[nr][nc] == background:
                                grid[nr][nc] = pattern_color
    
    return grid
