#!/usr/bin/env python3
import json

# Load task data
with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
    task = json.load(f)

def debug_pattern():
    """Debug to understand the exact transformation pattern"""
    train_pairs = task['train']
    
    for i, pair in enumerate(train_pairs):
        print(f"\n{'='*60}")
        print(f"DEBUG ANALYSIS - TRAIN PAIR {i+1}")
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
        
        # Find pattern cells
        pattern_cells = set()
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != bg_color:
                    pattern_cells.add((r, c))
        
        # Find changed cells
        changed_cells = set()
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != output_grid[r][c]:
                    changed_cells.add((r, c))
        
        if not pattern_cells:
            continue
            
        min_r = min(r for r, c in pattern_cells)
        max_r = max(r for r, c in pattern_cells)
        min_c = min(c for r, c in pattern_cells)
        max_c = max(c for r, c in pattern_cells)
        
        print(f"Pattern bbox: ({min_r},{min_c}) to ({max_r},{max_c})")
        print(f"Changed cells: {len(changed_cells)}")
        
        # Let me try a different approach: look at filled convex hull or rectangle
        # Find what areas are "inside" the pattern
        
        # Simple approach: for each row in the pattern, find the leftmost and rightmost pattern cells
        pattern_by_row = {}
        for r, c in pattern_cells:
            if r not in pattern_by_row:
                pattern_by_row[r] = []
            pattern_by_row[r].append(c)
        
        print("\nPattern spans by row:")
        for r in sorted(pattern_by_row.keys()):
            cols = sorted(pattern_by_row[r])
            print(f"  Row {r}: cols {min(cols)} to {max(cols)} (pattern at {cols})")
        
        print("\nAnalyzing changed cells row by row:")
        changed_by_row = {}
        for r, c in changed_cells:
            if r not in changed_by_row:
                changed_by_row[r] = []
            changed_by_row[r].append(c)
        
        for r in sorted(changed_by_row.keys()):
            cols = sorted(changed_by_row[r])
            print(f"  Row {r}: changed at cols {cols}")
            
            # For this row, what's the pattern span?
            if r in pattern_by_row:
                pattern_cols = sorted(pattern_by_row[r])
                pattern_min = min(pattern_cols) 
                pattern_max = max(pattern_cols)
                print(f"    Pattern in row {r}: cols {pattern_min}-{pattern_max}")
                
                # Check if changed cells are in "gaps" within pattern span
                for c in cols:
                    if pattern_min <= c <= pattern_max:
                        print(f"      ({r},{c}) is within pattern column span")
                    else:
                        print(f"      ({r},{c}) is outside pattern column span")
        
        # Let me also check the "convex hull" or filled rectangle approach
        print(f"\nTesting 'fill rectangle gaps' hypothesis:")
        predicted_fills = set()
        
        for r in range(min_r, max_r + 1):
            if r in pattern_by_row:
                pattern_cols = pattern_by_row[r]
                left_col = min(pattern_cols)
                right_col = max(pattern_cols)
                
                # Fill all background cells between leftmost and rightmost pattern
                for c in range(left_col, right_col + 1):
                    if input_grid[r][c] == bg_color:
                        predicted_fills.add((r, c))
        
        correct_fills = predicted_fills & changed_cells
        missed_fills = changed_cells - predicted_fills  
        extra_fills = predicted_fills - changed_cells
        
        print(f"Fill rectangle prediction: {len(correct_fills)}/{len(changed_cells)} correct")
        print(f"Missed: {missed_fills}")
        print(f"Extra: {extra_fills}")

if __name__ == "__main__":
    debug_pattern()