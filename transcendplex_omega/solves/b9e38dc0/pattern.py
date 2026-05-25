import json

with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b9e38dc0.json', 'r') as f:
    data = json.load(f)

example = data['train'][0]
input_grid = example['input']
output_grid = example['output']

print("Where do 9s appear in the output?")
print("(Looking for pattern in flooded areas)")

# Find regions that stayed as 1 in the output
stayed_1 = []
became_9 = []

for r in range(len(input_grid)):
    for c in range(len(input_grid[0])):
        if input_grid[r][c] == 1:
            if output_grid[r][c] == 1:
                stayed_1.append((r, c))
            elif output_grid[r][c] == 9:
                became_9.append((r, c))

print(f"Total cells that were background (1): {len(stayed_1) + len(became_9)}")
print(f"Stayed as 1: {len(stayed_1)}")
print(f"Became 9: {len(became_9)}")

print(f"\nCells that stayed as 1: {stayed_1}")

# Look at the spatial distribution
print("\nVisualizing what stayed as 1 vs became 9:")
for r in range(len(output_grid)):
    row_vis = ""
    for c in range(len(output_grid[0])):
        if input_grid[r][c] == 1:
            if output_grid[r][c] == 1:
                row_vis += "."  # Stayed background
            else:
                row_vis += "X"  # Became flood
        elif input_grid[r][c] == 3:
            row_vis += "B"  # Boundary
        elif input_grid[r][c] == 5:
            row_vis += "S"  # Special (5)
        elif input_grid[r][c] == 9:
            row_vis += "M"  # Marker
        else:
            row_vis += "?"
    print(f"Row {r:2}: {row_vis}")

print("\nLegend:")
print("  . = Background that stayed background")
print("  X = Background that became flooded")
print("  B = Boundary (3)")
print("  S = Special (5)")  
print("  M = Marker (9)")