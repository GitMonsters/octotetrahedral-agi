import json

# Load and analyze the puzzle
with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b9e38dc0.json', 'r') as f:
    data = json.load(f)

for i, example in enumerate(data['train']):
    print(f"\n=== Training Example {i+1} ===")
    input_grid = example['input']
    output_grid = example['output']
    
    print("INPUT:")
    for row in input_grid:
        print(row)
    
    print("\nOUTPUT:")
    for row in output_grid:
        print(row)
        
    # Find what changed
    height, width = len(input_grid), len(input_grid[0])
    changes = []
    for r in range(height):
        for c in range(width):
            if input_grid[r][c] != output_grid[r][c]:
                changes.append((r, c, input_grid[r][c], output_grid[r][c]))
    
    print(f"\nChanges (row, col, from, to): {len(changes)} changes")
    if len(changes) < 50:  # Only print if not too many
        for change in changes[:10]:  # First 10 changes
            print(f"  {change}")
        if len(changes) > 10:
            print(f"  ... and {len(changes) - 10} more")