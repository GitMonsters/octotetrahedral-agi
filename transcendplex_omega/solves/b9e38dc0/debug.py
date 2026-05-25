import json

# Debug specific example
with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b9e38dc0.json', 'r') as f:
    data = json.load(f)

example = data['train'][0]
input_grid = example['input']
output_grid = example['output']

print("=== Example 1 Analysis ===")
print("Background appears to be 1")
print("Other colors in input:")

# Find all colors in input
colors = set()
for row in input_grid:
    for cell in row:
        colors.add(cell)
print(f"Colors: {sorted(colors)}")

# Find positions of each color
color_positions = {}
for r in range(len(input_grid)):
    for c in range(len(input_grid[0])):
        color = input_grid[r][c]
        if color not in color_positions:
            color_positions[color] = []
        color_positions[color].append((r, c))

for color in sorted(colors):
    if color != 1:  # Skip background
        positions = color_positions[color]
        print(f"Color {color}: {len(positions)} positions - {positions[:5]}{'...' if len(positions) > 5 else ''}")

print("\nLooking at what happened in output:")
print("What cells changed from 1 to 9:")
changes_to_9 = []
for r in range(len(input_grid)):
    for c in range(len(input_grid[0])):
        if input_grid[r][c] == 1 and output_grid[r][c] == 9:
            changes_to_9.append((r, c))

print(f"Found {len(changes_to_9)} cells that changed from 1 to 9")
print(f"First few: {changes_to_9[:10]}")

# Check if color 9 stayed where it was
print(f"\nColor 9 in input: {color_positions.get(9, [])}")
if 9 in color_positions:
    r, c = color_positions[9][0]
    print(f"Color 9 position ({r},{c}) in output: {output_grid[r][c]}")