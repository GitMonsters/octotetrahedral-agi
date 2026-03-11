import json

with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b9e38dc0.json', 'r') as f:
    data = json.load(f)

example = data['train'][0]
input_grid = example['input']

print("Examining the 3-boundary structure:")
print("Where are the 3s?")

for r in range(len(input_grid)):
    for c in range(len(input_grid[0])):
        if input_grid[r][c] == 3:
            print(f"3 at ({r},{c})")

print("\nLet's see if there's a path from (4,8) to (0,0) that doesn't cross a 3:")

def can_reach(start_r, start_c, target_r, target_c, grid):
    """Check if we can reach target from start without crossing 3s"""
    height, width = len(grid), len(grid[0])
    stack = [(start_r, start_c)]
    visited = set()
    
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        if r < 0 or r >= height or c < 0 or c >= width:
            continue
        if grid[r][c] == 3:  # Hit boundary
            continue
            
        visited.add((r, c))
        
        if r == target_r and c == target_c:
            return True  # Found path!
            
        # Add neighbors
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited:
                stack.append((nr, nc))
    
    return False

# Check if marker can reach top-left corner
marker_pos = (4, 8)
can_reach_corner = can_reach(marker_pos[0], marker_pos[1], 0, 0, input_grid)
print(f"\nCan reach (0,0) from marker {marker_pos}: {can_reach_corner}")

# Check if marker can reach some points that should NOT be flooded
test_points = [(0, 0), (1, 1), (2, 2), (12, 7), (13, 8)]
for tr, tc in test_points:
    reachable = can_reach(marker_pos[0], marker_pos[1], tr, tc, input_grid)
    print(f"Can reach ({tr},{tc}) from marker: {reachable}")