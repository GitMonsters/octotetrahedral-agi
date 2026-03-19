"""
ARC Puzzle 54b177b5 Solver

Pattern: There's a template/stamp pattern in the background. Rectangular regions
of uniform color contain marker pixels. The template gets stamped onto each region,
centered on the marker pixel.
"""

def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    output = grid.copy()
    
    # Find the background color (most common)
    colors, counts = np.unique(grid, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Find connected rectangular regions of non-background uniform color
    visited = np.zeros_like(grid, dtype=bool)
    regions = []  # list of (region_color, min_r, max_r, min_c, max_c, marker_pos, marker_color)
    
    def flood_fill_rect(r, c, color):
        """Find rectangular region of given color starting from (r,c)"""
        # Find bounds of rectangular region
        min_r, max_r = r, r
        min_c, max_c = c, c
        
        # Expand to find the full rectangle
        # Check rows above and below
        while min_r > 0 and grid[min_r-1, c] != bg_color:
            min_r -= 1
        while max_r < h-1 and grid[max_r+1, c] != bg_color:
            max_r += 1
        while min_c > 0 and grid[r, min_c-1] != bg_color:
            min_c -= 1
        while max_c < w-1 and grid[r, max_c+1] != bg_color:
            max_c += 1
            
        # Now find the actual rectangle bounds by checking all cells
        # Find all non-bg connected cells
        cells = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in cells:
                continue
            if cr < 0 or cr >= h or cc < 0 or cc >= w:
                continue
            if grid[cr, cc] == bg_color:
                continue
            cells.add((cr, cc))
            stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])
        
        if not cells:
            return None
            
        min_r = min(cr for cr, cc in cells)
        max_r = max(cr for cr, cc in cells)
        min_c = min(cc for cr, cc in cells)
        max_c = max(cc for cr, cc in cells)
        
        return min_r, max_r, min_c, max_c, cells
    
    # Find all distinct non-background regions
    region_list = []
    checked = np.zeros_like(grid, dtype=bool)
    
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg_color and not checked[r, c]:
                result = flood_fill_rect(r, c, grid[r, c])
                if result:
                    min_r, max_r, min_c, max_c, cells = result
                    for cr, cc in cells:
                        checked[cr, cc] = True
                    region_list.append((min_r, max_r, min_c, max_c, cells))
    
    # Analyze each region to find:
    # 1. The template (small region with multiple colors)
    # 2. Target regions (larger, mostly uniform with a marker)
    
    def analyze_region(min_r, max_r, min_c, max_c, cells):
        """Analyze a region to determine its type and contents"""
        region_grid = grid[min_r:max_r+1, min_c:max_c+1]
        colors_in_region = {}
        for cr, cc in cells:
            color = grid[cr, cc]
            if color not in colors_in_region:
                colors_in_region[color] = []
            colors_in_region[color].append((cr, cc))
        return colors_in_region, region_grid
    
    # Find template and target regions
    template = None
    template_center_color = None
    template_pattern = None
    target_regions = []
    
    for min_r, max_r, min_c, max_c, cells in region_list:
        colors_in_region, region_grid = analyze_region(min_r, max_r, min_c, max_c, cells)
        
        # Count distinct colors
        num_colors = len(colors_in_region)
        region_size = len(cells)
        
        # Template has many colors relative to size, target regions are mostly uniform
        if num_colors >= 3 and region_size < 50:
            # This is likely the template
            template = (min_r, max_r, min_c, max_c, cells, colors_in_region)
        else:
            # Check if this is a target region (mostly one color with a marker)
            # Find the dominant color
            dominant_color = max(colors_in_region.keys(), key=lambda c: len(colors_in_region[c]))
            dominant_count = len(colors_in_region[dominant_color])
            
            # Check for markers (other colors)
            markers = []
            for color, positions in colors_in_region.items():
                if color != dominant_color and len(positions) == 1:
                    markers.append((color, positions[0]))
            
            if markers:
                target_regions.append((min_r, max_r, min_c, max_c, cells, dominant_color, markers))
    
    if template is None:
        return output.tolist()
    
    # Extract template pattern relative to its center marker
    t_min_r, t_max_r, t_min_c, t_max_c, t_cells, t_colors = template
    
    # Find the center of the template (the unique marker color that appears once)
    # The center is typically a color that appears exactly once
    center_pos = None
    center_color = None
    
    for color, positions in t_colors.items():
        if len(positions) == 1:
            # Check if this is surrounded by other pattern colors
            cr, cc = positions[0]
            # This could be the center
            if center_pos is None:
                center_pos = (cr, cc)
                center_color = color
            else:
                # Multiple single-pixel colors, need to pick the right one
                # The center is usually surrounded by other pattern colors
                pass
    
    # Build template pattern as offsets from center
    template_offsets = []
    for color, positions in t_colors.items():
        for pos in positions:
            dr = pos[0] - center_pos[0]
            dc = pos[1] - center_pos[1]
            template_offsets.append((dr, dc, color))
    
    # Apply template to each target region
    for min_r, max_r, min_c, max_c, cells, dominant_color, markers in target_regions:
        for marker_color, (mr, mc) in markers:
            # Stamp the template centered at (mr, mc)
            for dr, dc, color in template_offsets:
                nr, nc = mr + dr, mc + dc
                # Only apply within the region bounds
                if (nr, nc) in cells:
                    output[nr, nc] = color
    
    # Clear the original template from output
    for color, positions in t_colors.items():
        for pos in positions:
            output[pos[0], pos[1]] = bg_color
    
    return output.tolist()


if __name__ == "__main__":
    import json
    
    # Load the task
    task = json.load(open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json'))['54b177b5']
    
    # Test on all training examples
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            import numpy as np
            result_np = np.array(result)
            expected_np = np.array(expected)
            diff = result_np != expected_np
            diff_count = np.sum(diff)
            print(f"  Differences: {diff_count}")
            # Show first few differences
            diff_positions = np.argwhere(diff)[:5]
            for pos in diff_positions:
                r, c = pos
                print(f"    ({r},{c}): got {result_np[r,c]}, expected {expected_np[r,c]}")
