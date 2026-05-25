import json
with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/33b52de3.json") as f:
    task = json.load(f)

# Check second training example
example = task["train"][1]
input_grid = example["input"]
output_grid = example["output"]

print("=== SECOND TRAINING EXAMPLE ===")

# Find key pattern
print("Key pattern:")
for i in range(3, 7):  # Key seems to be at different position
    row_vals = []
    for j in range(1, 6):
        val = input_grid[i][j]
        if val != 0:
            row_vals.append(val)
        else:
            row_vals.append('.')
    print(f"Row {i}: {row_vals}")

# Find the actual key bounds
key_positions = []
for i in range(len(input_grid)):
    for j in range(len(input_grid[0])):
        if input_grid[i][j] != 0 and input_grid[i][j] != 5:
            key_positions.append((i, j, input_grid[i][j]))

print(f"\nKey positions: {key_positions}")

# Check the first few blocks in the output
print("\nFirst block output (rows 1-3, cols 6-8):")
for i in range(1, 4):
    for j in range(6, 9):
        if j < len(output_grid[0]) and i < len(output_grid):
            print(f"({i},{j}): {output_grid[i][j]}")

print("\nSecond block output (rows 1-3, cols 10-12):")  
for i in range(1, 4):
    for j in range(10, 13):
        if j < len(output_grid[0]) and i < len(output_grid):
            print(f"({i},{j}): {output_grid[i][j]}")