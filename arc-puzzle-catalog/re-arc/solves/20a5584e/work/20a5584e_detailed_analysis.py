import json
import numpy as np

def analyze_patterns_detailed():
    """Detailed analysis focusing on the actual shapes and transformations"""
    
    with open('/Users/evanpieser/apr12_tasks/20a5584e.json', 'r') as f:
        task = json.load(f)
    
    for i, pair in enumerate(task['train']):
        print(f"\n=== TRAINING PAIR {i+1} DETAILED ===")
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        height, width = input_grid.shape
        
        # Find background color (most frequent)
        from collections import Counter
        color_counts = Counter(input_grid.flatten())
        background = max(color_counts, key=color_counts.get)
        
        # Find all non-background colors
        all_colors = set(input_grid.flatten())
        non_bg_colors = all_colors - {background}
        
        print(f"Background: {background}, Non-background colors: {sorted(non_bg_colors)}")
        
        # Find existing shape/pattern (the non-background, non-1 colors)
        pattern_colors = non_bg_colors - {1}
        print(f"Pattern colors (not 1, not background): {sorted(pattern_colors)}")
        
        if pattern_colors:
            # Find the original shape made of pattern colors
            original_shape_cells = []
            for pattern_color in pattern_colors:
                for r in range(height):
                    for c in range(width):
                        if input_grid[r, c] == pattern_color:
                            original_shape_cells.append((r, c, pattern_color))
            
            print(f"Original shape cells: {original_shape_cells}")
            
            # Calculate bounding box of original shape
            if original_shape_cells:
                shape_rows = [cell[0] for cell in original_shape_cells]
                shape_cols = [cell[1] for cell in original_shape_cells]
                min_r, max_r = min(shape_rows), max(shape_rows)
                min_c, max_c = min(shape_cols), max(shape_cols)
                print(f"Original shape bounding box: ({min_r}, {min_c}) to ({max_r}, {max_c})")
                
                # Create relative pattern from bounding box
                relative_pattern = []
                for r, c, color in original_shape_cells:
                    relative_pattern.append((r - min_r, c - min_c, color))
                print(f"Relative pattern (dr, dc, color): {relative_pattern}")
        
        # Find positions of 1s (blue dots) in input
        ones_in_input = []
        for r in range(height):
            for c in range(width):
                if input_grid[r, c] == 1:
                    ones_in_input.append((r, c))
        
        # Find positions of 1s that are isolated (surrounded by background)
        isolated_ones = []
        for r, c in ones_in_input:
            is_isolated = True
            # Check 8-neighborhood
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if input_grid[nr, nc] != background and input_grid[nr, nc] != 1:
                            is_isolated = False
                            break
                if not is_isolated:
                    break
            if is_isolated:
                isolated_ones.append((r, c))
        
        print(f"Isolated 1s: {isolated_ones}")
        
        # Find what new shapes appear in the output around these isolated 1s
        new_shapes = []
        for r in range(height):
            for c in range(width):
                if input_grid[r, c] != output_grid[r, c]:
                    new_shapes.append((r, c, input_grid[r, c], output_grid[r, c]))
        
        print(f"New shapes (first 10): {new_shapes[:10]}")
        
        # Group new shapes by isolated 1 positions
        if isolated_ones and new_shapes:
            print(f"\nAnalyzing new shapes around isolated 1s:")
            for iso_r, iso_c in isolated_ones:
                nearby_changes = []
                for r, c, old_color, new_color in new_shapes:
                    # Check if this change is near this isolated 1
                    if abs(r - iso_r) <= 3 and abs(c - iso_c) <= 3:  # Within 3 cells
                        nearby_changes.append((r - iso_r, c - iso_c, old_color, new_color))
                print(f"  Around ({iso_r}, {iso_c}): {nearby_changes}")

if __name__ == "__main__":
    analyze_patterns_detailed()