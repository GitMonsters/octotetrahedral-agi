import json

# Let's trace the exact flood pattern in example 1
with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b9e38dc0.json', 'r') as f:
    data = json.load(f)

example = data['train'][0]
input_grid = example['input']
output_grid = example['output']

print("Input grid around the 9:")
for r in range(max(0, 4-2), min(len(input_grid), 4+3)):
    row = ""
    for c in range(max(0, 8-5), min(len(input_grid[0]), 8+6)):
        row += str(input_grid[r][c])
    print(f"Row {r}: {row}")

print("\nOutput grid same area:")
for r in range(max(0, 4-2), min(len(output_grid), 4+3)):
    row = ""
    for c in range(max(0, 8-5), min(len(output_grid[0]), 8+6)):
        row += str(output_grid[r][c])
    print(f"Row {r}: {row}")

print("\nLet's see the bottom area around the 5s:")
print("Input grid around the 5s (row 11):")
for r in range(9, 15):
    row = ""
    for c in range(5, 12):
        row += str(input_grid[r][c])
    print(f"Row {r}: {row}")

print("\nOutput grid same area:")
for r in range(9, 15):
    row = ""
    for c in range(5, 12):
        if r < len(output_grid):
            row += str(output_grid[r][c])
        else:
            row += "?"
    print(f"Row {r}: {row}")