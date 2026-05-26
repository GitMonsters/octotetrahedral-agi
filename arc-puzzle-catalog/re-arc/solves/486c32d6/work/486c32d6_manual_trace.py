#!/usr/bin/env python3
import json

# Load task
with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
    task = json.load(f)

print("=== MANUAL TRACE - TRAIN 0 ROW 1 ===")

example = task['train'][0]
input_row = example['input'][1]
output_row = example['output'][1]

print(f"Input:  {input_row}")
print(f"Output: {output_row}")

# Known separators for train 0: [3, 7, 11, 15, 19]
separators = [3, 7, 11, 15, 19]
print(f"Separators: {separators}")

# Define cells
cells = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 23)]
print(f"Cells: {cells}")

print("\nCell contents (excluding separators):")
for i, (start, end) in enumerate(cells):
    content_in = []
    content_out = []
    for c in range(start, end):
        if c not in separators:
            content_in.append(input_row[c])
            content_out.append(output_row[c])
    print(f"Cell {i}: input={content_in}, output={content_out}")

print("\nObservation:")
print("- Cell 0: [2, 2, 5] -> [2, 2, 5] (unchanged)")
print("- Cell 1: [2, 2, 2] -> [2, 2, 5] (pos 2: 2->5)")
print("- Cell 2: [2, 2, 2] -> [2, 2, 5] (pos 2: 2->5)")
print("- Cell 3: [2, 2, 2] -> [2, 2, 5] (pos 2: 2->5)")
print("- Cell 4: [2, 2, 5] -> [2, 2, 5] (unchanged)")
print("- Cell 5: [2, 2, 2] -> [2, 2, 2] (unchanged - last cell special?)")

print("\nPattern: Value 5 at position 2 exists in cells 0 and 4")
print("         It should propagate to cells 1, 2, 3 but NOT cell 5")
print("         Why not cell 5? Maybe because it's incomplete/partial cell?")

print(f"\nCell 5 has only {23-20} positions vs others have {4-1} positions")

print("\n=== TRAIN 0 ROW 16 ===")
input_row = example['input'][16]  
output_row = example['output'][16]

print(f"Input:  {input_row}")
print(f"Output: {output_row}")

print("\nCell contents (excluding separators):")
for i, (start, end) in enumerate(cells):
    content_in = []
    content_out = []
    for c in range(start, end):
        if c not in separators:
            content_in.append(input_row[c])
            content_out.append(output_row[c])
    print(f"Cell {i}: input={content_in}, output={content_out}")

print("\nObservation:")
print("- Value 1 at position 1 exists in cells 0, 2, 5")
print("- It propagates to cells 1, 3, 4 at position 1")
print("- ALL cells get the anomaly this time, including cell 5")