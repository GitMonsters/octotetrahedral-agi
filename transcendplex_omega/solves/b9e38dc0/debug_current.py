import sys
import json

def solve(grid):
    result = [row[:] for row in grid]
    height, width = len(grid), len(grid[0])
    
    # Find the background color (most common)
    color_counts = {}
    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1
    background = max(color_counts, key=color_counts.get)
    
    # Find all non-background colors and their positions
    special_colors = {}
    for r in range(height):
        for c in range(width):
            if grid[r][c] != background:
                color = grid[r][c]
                if color not in special_colors:
                    special_colors[color] = []
                special_colors[color].append((r, c))
    
    # Find the flood color (single isolated cell = marker)
    flood_color = None
    marker_pos = None
    for color, positions in special_colors.items():
        if len(positions) == 1:
            flood_color = color
            marker_pos = positions[0]
            break
    
    if flood_color is None:
        return result
    
    # Find boundary colors (form structural barriers)  
    boundary_colors = set()
    secondary_sources = []
    
    for color, positions in special_colors.items():
        if color != flood_color:
            if len(positions) > 5:  # Large structures = boundaries
                boundary_colors.add(color)
            else:  # Small clusters = secondary flood sources
                secondary_sources.extend(positions)
    
    # Flood fill with boundaries
    def flood_fill_with_boundaries(start_positions, target_color, boundaries):
        stack = start_positions[:]
        visited = set()
        
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or r < 0 or r >= height or c < 0 or c >= width:
                continue
                
            current = result[r][c]
            
            # Don't flood over boundary colors
            if current in boundaries:
                continue
                
            visited.add((r, c))
            
            # Fill background or target color cells
            if current == background or current == target_color:
                result[r][c] = target_color
                
                # Add neighbors
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in visited:
                        stack.append((nr, nc))
    
    # Start flood from marker position
    flood_fill_with_boundaries([marker_pos], flood_color, boundary_colors)
    
    # Start flood from secondary sources (if any)
    if secondary_sources:
        flood_fill_with_boundaries(secondary_sources, flood_color, boundary_colors)
    
    return result

# Test
with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b9e38dc0.json', 'r') as f:
    data = json.load(f)

example = data['train'][0]
input_grid = example['input']
expected = example['output']
result = solve(input_grid)

print("MY RESULT:")
for r, row in enumerate(result):
    vis = ""
    for c, cell in enumerate(row):
        if cell == 1:
            vis += "."
        elif cell == 3:
            vis += "B"
        elif cell == 5:
            vis += "S"
        elif cell == 9:
            vis += "X"
        else:
            vis += str(cell)
    print(f"Row {r:2}: {vis}")

print("\nEXPECTED:")
for r, row in enumerate(expected):
    vis = ""
    for c, cell in enumerate(row):
        if cell == 1:
            vis += "."
        elif cell == 3:
            vis += "B"
        elif cell == 5:
            vis += "S"
        elif cell == 9:
            vis += "X"
        else:
            vis += str(cell)
    print(f"Row {r:2}: {vis}")

matches = sum(1 for r in range(len(input_grid)) for c in range(len(input_grid[0])) 
             if result[r][c] == expected[r][c])
total = len(input_grid) * len(input_grid[0])
print(f"\nMatches: {matches}/{total} = {matches/total:.2%}")