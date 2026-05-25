import json
import sys

# Load the puzzle
with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/c7f57c3e.json', 'r') as f:
    task = json.load(f)

def print_diff(input_grid, output_grid, title):
    print(f"\n=== {title} ===")
    rows, cols = len(input_grid), len(input_grid[0])
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != output_grid[r][c]:
                print(f"({r},{c}): {input_grid[r][c]} -> {output_grid[r][c]}")

for i, example in enumerate(task['train']):
    print_diff(example['input'], example['output'], f"Training Example {i+1}")

print_diff(task['test'][0]['input'], task['test'][0]['output'], "Test Example")