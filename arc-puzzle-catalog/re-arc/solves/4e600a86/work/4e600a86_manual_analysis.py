#!/usr/bin/env python3
import json

# Load task data  
with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
    task = json.load(f)

def manual_analysis():
    """Manual inspection of the grids to understand the pattern"""
    train_pairs = task['train']
    
    for i, pair in enumerate(train_pairs):
        print(f"\n{'='*60}")
        print(f"MANUAL ANALYSIS - TRAIN PAIR {i+1}")
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
            print("Background is already 3 - no change")
            continue
        
        # Print the grids side by side for manual inspection
        print("\nINPUT GRID:")
        for r in range(h):
            row_str = ""
            for c in range(w):
                if input_grid[r][c] == bg_color:
                    row_str += ". "
                else:
                    row_str += f"{input_grid[r][c]} "
            print(f"{r:2d}: {row_str}")
        
        print("\nOUTPUT GRID:")
        for r in range(h):
            row_str = ""
            for c in range(w):
                if output_grid[r][c] == bg_color:
                    row_str += ". "
                elif output_grid[r][c] == 3:
                    row_str += "3 "
                else:
                    row_str += f"{output_grid[r][c]} "
            print(f"{r:2d}: {row_str}")
        
        # Show changes
        print("\nCHANGES (background cells that became 3):")
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != output_grid[r][c]:
                    print(f"  ({r},{c}): {input_grid[r][c]} → {output_grid[r][c]}")
        
        # Find pattern cells
        pattern_cells = set()
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != bg_color:
                    pattern_cells.add((r, c))
        
        if pattern_cells:
            min_r = min(r for r, c in pattern_cells)
            max_r = max(r for r, c in pattern_cells)
            min_c = min(c for r, c in pattern_cells)
            max_c = max(c for r, c in pattern_cells)
            
            print(f"\nPattern bounding box: ({min_r},{min_c}) to ({max_r},{max_c})")
            
            # Let me try a different approach: Look for "enclosed" background areas
            # A background cell might be changed to 3 if it's "surrounded" by pattern cells
            
            changed_cells = set()
            for r in range(h):
                for c in range(w):
                    if input_grid[r][c] != output_grid[r][c]:
                        changed_cells.add((r, c))
            
            print(f"\nAnalyzing 'enclosure' hypothesis:")
            for r, c in changed_cells:
                # Check neighbors
                neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
                pattern_neighbors = 0
                for nr, nc in neighbors:
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) in pattern_cells:
                        pattern_neighbors += 1
                
                print(f"  ({r},{c}): {pattern_neighbors}/4 pattern neighbors")
            
            # Try another hypothesis: flood fill from pattern edges
            print(f"\nAnalyzing 'flood fill' hypothesis:")
            # Find which background cells are "internal" to the pattern
            
            # For each changed cell, find the nearest pattern cells
            for r, c in changed_cells:
                distances_to_pattern = []
                for pr, pc in pattern_cells:
                    dist = abs(r - pr) + abs(c - pc)  # Manhattan distance
                    distances_to_pattern.append(dist)
                
                if distances_to_pattern:
                    min_dist = min(distances_to_pattern)
                    print(f"  ({r},{c}): min distance to pattern = {min_dist}")

if __name__ == "__main__":
    manual_analysis()