import json

# Simple test - just try different flood strategies on example 1
with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b9e38dc0.json', 'r') as f:
    data = json.load(f)

def simple_flood_everywhere(grid):
    """Just flood everywhere with color 9 except where there's color 3"""
    result = [row[:] for row in grid]
    height, width = len(grid), len(grid[0])
    
    for r in range(height):
        for c in range(width):
            if result[r][c] == 1:  # Background
                result[r][c] = 9  # Flood color
            # Keep 3s and 5s as they are
    
    return result

def simple_flood_from_marker(grid):
    """Flood from the 9 position only"""
    result = [row[:] for row in grid]
    height, width = len(grid), len(grid[0])
    
    # Find the 9
    start_r, start_c = None, None
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 9:
                start_r, start_c = r, c
                break
        if start_r is not None:
            break
    
    if start_r is None:
        return result
    
    # Simple flood fill
    stack = [(start_r, start_c)]
    visited = set()
    
    while stack:
        r, c = stack.pop()
        if (r, c) in visited or r < 0 or r >= height or c < 0 or c >= width:
            continue
        if result[r][c] not in [1, 9]:  # Don't flood over barriers
            continue
            
        visited.add((r, c))
        result[r][c] = 9
        
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited:
                stack.append((nr, nc))
    
    return result

# Test on example 1
example = data['train'][0]
input_grid = example['input']
expected = example['output']

print("Testing strategy 1: Flood everywhere except barriers")
result1 = simple_flood_everywhere(input_grid)
matches1 = sum(1 for r in range(len(input_grid)) for c in range(len(input_grid[0])) 
              if result1[r][c] == expected[r][c])
total = len(input_grid) * len(input_grid[0])
print(f"Matches: {matches1}/{total} = {matches1/total:.2%}")

print("\nTesting strategy 2: Flood from marker only")
result2 = simple_flood_from_marker(input_grid)
matches2 = sum(1 for r in range(len(input_grid)) for c in range(len(input_grid[0])) 
              if result2[r][c] == expected[r][c])
print(f"Matches: {matches2}/{total} = {matches2/total:.2%}")

# Check where strategy 2 differs from expected
print("\nWhere strategy 2 differs from expected:")
diffs = 0
for r in range(len(input_grid)):
    for c in range(len(input_grid[0])):
        if result2[r][c] != expected[r][c]:
            if diffs < 10:  # Show first 10 differences
                print(f"({r},{c}): got {result2[r][c]}, expected {expected[r][c]}")
            diffs += 1
print(f"Total differences: {diffs}")