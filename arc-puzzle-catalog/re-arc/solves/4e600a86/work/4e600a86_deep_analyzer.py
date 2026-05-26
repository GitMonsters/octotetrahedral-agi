#!/usr/bin/env python3
import json

# Load task data
with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
    task = json.load(f)

def detailed_analysis():
    """Deep analysis of the transformation pattern"""
    train_pairs = task['train']
    
    for i, pair in enumerate(train_pairs):
        print(f"\n{'='*50}")
        print(f"DETAILED ANALYSIS - TRAIN PAIR {i+1}")
        print(f"{'='*50}")
        
        input_grid = pair['input']
        output_grid = pair['output']
        h, w = len(input_grid), len(input_grid[0])
        
        # Find background color
        colors = {}
        for row in input_grid:
            for cell in row:
                colors[cell] = colors.get(cell, 0) + 1
        bg_color = max(colors, key=colors.get)
        
        # Find pattern cells
        pattern_cells = set()
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != bg_color:
                    pattern_cells.add((r, c))
        
        # Find changed cells
        changed_cells = []
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != output_grid[r][c]:
                    changed_cells.append((r, c, input_grid[r][c], output_grid[r][c]))
        
        print(f"Background: {bg_color}")
        print(f"Pattern cells: {len(pattern_cells)}")
        print(f"Changed cells: {len(changed_cells)}")
        
        if len(changed_cells) == 0:
            print("No changes - background is already 3!")
            continue
            
        # Find what color changed cells become
        target_colors = set()
        for r, c, from_color, to_color in changed_cells:
            target_colors.add(to_color)
        print(f"Changed cells become: {target_colors}")
        
        # Check if all changes are bg_color → 3
        all_bg_to_3 = all(from_color == bg_color and to_color == 3 
                         for r, c, from_color, to_color in changed_cells)
        print(f"All changes are background→3: {all_bg_to_3}")
        
        if not pattern_cells:
            continue
            
        # Pattern bounding box
        min_r = min(r for r, c in pattern_cells)
        max_r = max(r for r, c in pattern_cells)
        min_c = min(c for r, c in pattern_cells)
        max_c = max(c for r, c in pattern_cells)
        
        print(f"Pattern bbox: ({min_r},{min_c}) to ({max_r},{max_c})")
        
        # Check which background cells within bbox got changed
        bg_cells_in_bbox = []
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if input_grid[r][c] == bg_color:
                    bg_cells_in_bbox.append((r, c))
        
        changed_pos = set((r, c) for r, c, _, _ in changed_cells)
        bg_changed_in_bbox = [pos for pos in bg_cells_in_bbox if pos in changed_pos]
        bg_not_changed_in_bbox = [pos for pos in bg_cells_in_bbox if pos not in changed_pos]
        
        print(f"Background cells in bbox: {len(bg_cells_in_bbox)}")
        print(f"Background cells changed in bbox: {len(bg_changed_in_bbox)}")
        print(f"Background cells NOT changed in bbox: {len(bg_not_changed_in_bbox)}")
        
        # Look for mirror/reflection patterns
        print("\nLooking for reflection patterns...")
        pattern_height = max_r - min_r + 1
        pattern_width = max_c - min_c + 1
        center_r = min_r + pattern_height // 2
        center_c = min_c + pattern_width // 2
        
        print(f"Pattern center: ({center_r}, {center_c})")
        
        # Check vertical reflection hypothesis
        print("\nChecking vertical reflection hypothesis:")
        vertical_reflection_matches = 0
        vertical_reflection_total = 0
        
        for r, c in bg_changed_in_bbox:
            # Find vertically reflected position
            reflected_r = center_r + (center_r - r)  # or 2*center_r - r
            
            if min_r <= reflected_r <= max_r:
                vertical_reflection_total += 1
                if (reflected_r, c) in pattern_cells:
                    vertical_reflection_matches += 1
                    print(f"  ({r},{c}) reflects to ({reflected_r},{c}) - HIT!")
                else:
                    print(f"  ({r},{c}) reflects to ({reflected_r},{c}) - miss")
        
        if vertical_reflection_total > 0:
            print(f"Vertical reflection accuracy: {vertical_reflection_matches}/{vertical_reflection_total} = {vertical_reflection_matches/vertical_reflection_total:.2%}")
        
        # Check horizontal reflection hypothesis  
        print("\nChecking horizontal reflection hypothesis:")
        horizontal_reflection_matches = 0
        horizontal_reflection_total = 0
        
        for r, c in bg_changed_in_bbox:
            # Find horizontally reflected position
            reflected_c = center_c + (center_c - c)  # or 2*center_c - c
            
            if min_c <= reflected_c <= max_c:
                horizontal_reflection_total += 1
                if (r, reflected_c) in pattern_cells:
                    horizontal_reflection_matches += 1
                    print(f"  ({r},{c}) reflects to ({r},{reflected_c}) - HIT!")
                else:
                    print(f"  ({r},{c}) reflects to ({r},{reflected_c}) - miss")
        
        if horizontal_reflection_total > 0:
            print(f"Horizontal reflection accuracy: {horizontal_reflection_matches}/{horizontal_reflection_total} = {horizontal_reflection_matches/horizontal_reflection_total:.2%}")

if __name__ == "__main__":
    detailed_analysis()