#!/usr/bin/env python3
import json

# Load task data
with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
    task = json.load(f)

def analyze_changes(input_grid, output_grid):
    """Find exactly which cells changed from input to output"""
    changes = []
    
    h, w = len(input_grid), len(input_grid[0])
    for r in range(h):
        for c in range(w):
            if input_grid[r][c] != output_grid[r][c]:
                changes.append({
                    'position': (r, c),
                    'from': input_grid[r][c],
                    'to': output_grid[r][c]
                })
    
    return changes

def analyze_all_examples():
    """Analyze all training examples"""
    train_pairs = task['train']
    
    for i, pair in enumerate(train_pairs):
        print(f"\n=== TRAIN PAIR {i+1} ===")
        input_grid = pair['input']
        output_grid = pair['output']
        
        # Find background color
        colors = {}
        for row in input_grid:
            for cell in row:
                colors[cell] = colors.get(cell, 0) + 1
        bg_color = max(colors, key=colors.get)
        print(f"Background color: {bg_color}")
        
        # Find pattern color(s)
        pattern_colors = set()
        for row in input_grid:
            for cell in row:
                if cell != bg_color:
                    pattern_colors.add(cell)
        print(f"Pattern colors: {pattern_colors}")
        
        # Analyze changes
        changes = analyze_changes(input_grid, output_grid)
        print(f"Number of cells changed: {len(changes)}")
        
        if changes:
            print("Changes:")
            for change in changes:
                r, c = change['position']
                print(f"  ({r}, {c}): {change['from']} → {change['to']}")
                
        # Find pattern cells and changed cells
        pattern_cells = set()
        for r in range(len(input_grid)):
            for c in range(len(input_grid[0])):
                if input_grid[r][c] != bg_color:
                    pattern_cells.add((r, c))
        
        changed_cells = set((change['position'][0], change['position'][1]) for change in changes)
        
        print(f"Pattern cells (non-background): {len(pattern_cells)}")
        print(f"Changed cells: {len(changed_cells)}")
        
        # Look for geometric relationships
        if pattern_cells and changed_cells:
            # Find bounding box of pattern
            min_r = min(r for r, c in pattern_cells)
            max_r = max(r for r, c in pattern_cells)
            min_c = min(c for r, c in pattern_cells)
            max_c = max(c for r, c in pattern_cells)
            
            print(f"Pattern bounding box: rows {min_r}-{max_r}, cols {min_c}-{max_c}")
            
            # Check if changed cells are within pattern bounding box
            within_bbox = [cell for cell in changed_cells 
                          if min_r <= cell[0] <= max_r and min_c <= cell[1] <= max_c]
            print(f"Changed cells within pattern bbox: {len(within_bbox)}")
            
            if within_bbox:
                print("Changed cells within bbox:")
                for r, c in within_bbox:
                    print(f"  ({r}, {c})")

if __name__ == "__main__":
    analyze_all_examples()