import json

with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b9e38dc0.json', 'r') as f:
    data = json.load(f)

# Example 2 analysis
example = data['train'][1]
input_grid = example['input']
output_grid = example['output']

print("=== Example 2 ===")
print("INPUT:")
for r, row in enumerate(input_grid):
    print(f"{r:2}: {''.join(str(x) for x in row)}")

print("\nOUTPUT:")
for r, row in enumerate(output_grid):
    print(f"{r:2}: {''.join(str(x) for x in row)}")

# Find colors
colors = set()
for row in input_grid:
    for cell in row:
        colors.add(cell)

color_positions = {}
for r in range(len(input_grid)):
    for c in range(len(input_grid[0])):
        color = input_grid[r][c]
        if color not in color_positions:
            color_positions[color] = []
        color_positions[color].append((r, c))

print(f"\nColors in input: {sorted(colors)}")
for color in sorted(colors):
    positions = color_positions[color]
    print(f"Color {color}: {len(positions)} positions")
    if len(positions) < 20:
        print(f"  Positions: {positions}")

# See what changed
print(f"\nWhat changed from 0 to 4:")
changes = []
for r in range(len(input_grid)):
    for c in range(len(input_grid[0])):
        if input_grid[r][c] == 0 and output_grid[r][c] == 4:
            changes.append((r, c))

print(f"Found {len(changes)} cells changed from 0 to 4")
print(f"Sample changes: {changes[:10]}")

# Check if 4 is preserved
if 4 in color_positions:
    pos = color_positions[4][0]
    print(f"Position of 4 in input: {pos}")
    print(f"Value at that position in output: {output_grid[pos[0]][pos[1]]}")