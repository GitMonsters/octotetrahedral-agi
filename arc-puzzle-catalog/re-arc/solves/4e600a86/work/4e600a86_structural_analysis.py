#!/usr/bin/env python3
import json

# Load task data
with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
    task = json.load(f)

def examine_structural_pattern():
    """Examine if there's a structural completion pattern"""
    train_pairs = task['train']
    
    for i, pair in enumerate(train_pairs):
        print(f"\n{'='*60}")
        print(f"STRUCTURAL ANALYSIS - TRAIN PAIR {i+1}")
        print(f"{'='*60}")
        
        input_grid = pair['input']
        output_grid = pair['output']
        h, w = len(input_grid), len(input_grid[0])
        
        # Find background color
        colors = {}
        for row in input_grid:
            for cell in row:
                colors[cell] = colors.get(cell, 0) + 1
        bg_color = max(colors, key=colors.get)
        
        if bg_color == 3:
            print("Background is already 3")
            continue
        
        # Create visual representation showing original pattern and changes
        print("\nOriginal pattern + changes (. = bg, X = pattern, 3 = added):")
        
        for r in range(h):
            if r < len(input_grid):
                row_str = f"{r:2d}: "
                for c in range(w):
                    if input_grid[r][c] != bg_color:
                        row_str += "X "  # Original pattern
                    elif output_grid[r][c] == 3:
                        row_str += "3 "  # Added in output
                    else:
                        row_str += ". "  # Background
                
                print(row_str)
        
        # Find pattern and changed positions
        pattern_cells = set()
        changed_cells = set()
        
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != bg_color:
                    pattern_cells.add((r, c))
                if input_grid[r][c] != output_grid[r][c]:
                    changed_cells.add((r, c))
        
        if not pattern_cells or not changed_cells:
            continue
        
        min_r = min(r for r, c in pattern_cells)
        max_r = max(r for r, c in pattern_cells)
        min_c = min(c for r, c in pattern_cells)
        max_c = max(c for r, c in pattern_cells)
        
        print(f"\nPattern bbox: ({min_r},{min_c}) to ({max_r},{max_c})")
        print(f"Changed cells: {len(changed_cells)}")
        
        # Let me examine if it's about creating a "left extension" or "mirrored completion"
        # Check if changed cells have a consistent spatial relationship to pattern
        
        print(f"\nSpatial relationship analysis:")
        
        for r, c in changed_cells:
            # Find closest pattern cell(s)
            min_dist = float('inf')
            closest_pattern_cells = []
            
            for pr, pc in pattern_cells:
                dist = abs(r - pr) + abs(c - pc)  # Manhattan distance
                if dist < min_dist:
                    min_dist = dist
                    closest_pattern_cells = [(pr, pc)]
                elif dist == min_dist:
                    closest_pattern_cells.append((pr, pc))
            
            # Analyze position relative to pattern
            relative_to_left = c < min_c
            relative_to_right = c > max_c
            relative_to_pattern_rows = min_r <= r <= max_r
            
            print(f"  ({r},{c}): dist={min_dist}, left_of_pattern={relative_to_left}, "
                  f"right_of_pattern={relative_to_right}, within_pattern_rows={relative_to_pattern_rows}")
        
        # Check if all changes are to the left of pattern
        all_changes_left = all(c < min_c for r, c in changed_cells)
        all_changes_within_rows = all(min_r <= r <= max_r for r, c in changed_cells)
        
        print(f"\nAll changes to left of pattern: {all_changes_left}")
        print(f"All changes within pattern row range: {all_changes_within_rows}")
        
        # Test another hypothesis: maybe it's about creating "left reflection" 
        # where we reflect pattern parts to create symmetry around a vertical axis
        
        # Find the leftmost column that could serve as reflection axis
        left_edge = min_c
        
        print(f"\nTesting left-reflection hypothesis (axis at col {left_edge-1}):")
        
        predicted_left_reflection = set()
        for r, c in pattern_cells:
            # Reflect across vertical line at left_edge - 1
            reflected_c = 2 * (left_edge - 1) - c
            if reflected_c >= 0 and reflected_c < left_edge:
                predicted_left_reflection.add((r, reflected_c))
        
        correct_left = predicted_left_reflection & changed_cells
        print(f"Left reflection matches: {len(correct_left)}/{len(changed_cells)}")

if __name__ == "__main__":
    examine_structural_pattern()