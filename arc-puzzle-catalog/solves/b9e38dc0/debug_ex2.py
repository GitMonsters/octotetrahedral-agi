import json

# Focus on example 2 since it's most accurate
with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b9e38dc0.json', 'r') as f:
    data = json.load(f)

def solve_current(grid):
    result = [row[:] for row in grid]
    height, width = len(grid), len(grid[0])
    
    color_counts = {}
    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1
    background = max(color_counts, key=color_counts.get)
    
    special_colors = {}
    for r in range(height):
        for c in range(width):
            if grid[r][c] != background:
                color = grid[r][c]
                if color not in special_colors:
                    special_colors[color] = []
                special_colors[color].append((r, c))
    
    flood_color = None
    marker_pos = None
    for color, positions in special_colors.items():
        if len(positions) == 1:
            flood_color = color
            marker_pos = positions[0]
            break
    
    if flood_color is None:
        return result
    
    secondary_sources = []
    for color, positions in special_colors.items():
        if color != flood_color and len(positions) <= 5:
            secondary_sources.extend(positions)
    
    def flood_fill(start_r, start_c, target_color):
        stack = [(start_r, start_c)]
        visited = set()
        
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or r < 0 or r >= height or c < 0 or c >= width:
                continue
            if result[r][c] != background and result[r][c] != target_color:
                continue
                
            visited.add((r, c))
            result[r][c] = target_color
            
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited:
                    stack.append((nr, nc))
    
    flood_fill(marker_pos[0], marker_pos[1], flood_color)
    for sr, sc in secondary_sources:
        flood_fill(sr, sc, flood_color)
    
    return result

example = data['train'][1]  # Example 2
input_grid = example['input']
expected = example['output']
result = solve_current(input_grid)

print("=== Example 2 Analysis ===")
print("MY RESULT:")
for r, row in enumerate(result):
    print(f"{r}: {''.join('.' if x==0 else 'X' if x==4 else str(x) for x in row)}")

print("\nEXPECTED:")
for r, row in enumerate(expected):
    print(f"{r}: {''.join('.' if x==0 else 'X' if x==4 else str(x) for x in row)}")

print("\nDIFF (R=right, W=wrong):")
for r in range(len(input_grid)):
    diff_line = ""
    for c in range(len(input_grid[0])):
        if result[r][c] == expected[r][c]:
            diff_line += "R"
        else:
            diff_line += "W"
    print(f"{r}: {diff_line}")

matches = sum(1 for r in range(len(input_grid)) for c in range(len(input_grid[0])) 
             if result[r][c] == expected[r][c])
total = len(input_grid) * len(input_grid[0])
print(f"\nAccuracy: {matches}/{total} = {matches/total:.1%}")