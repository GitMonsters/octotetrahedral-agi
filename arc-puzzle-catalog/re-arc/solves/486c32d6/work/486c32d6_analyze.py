import json

# Load task data
with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
    task_data = json.load(f)

print("=== MANUAL ANALYSIS ===")

for i, example in enumerate(task_data['train']):
    print(f"\nTRAIN {i}:")
    input_grid = example['input']
    output_grid = example['output'] 
    
    print(f"Size: {len(input_grid)}x{len(input_grid[0])}")
    
    # Find rows with changes
    changed_rows = []
    for r in range(len(input_grid)):
        if input_grid[r] != output_grid[r]:
            changed_rows.append(r)
    
    print(f"Changed rows: {changed_rows}")
    
    # Analyze specific changes for first few changed rows
    for r in changed_rows[:3]:
        print(f"\nRow {r}:")
        print(f"Input:  {input_grid[r]}")
        print(f"Output: {output_grid[r]}")
        
        # Find anomalies in input row
        from collections import Counter
        counter = Counter(input_grid[r])
        print(f"Value counts: {dict(counter)}")
        
        # Find differences
        diffs = []
        for c in range(len(input_grid[r])):
            if input_grid[r][c] != output_grid[r][c]:
                diffs.append(f"Col {c}: {input_grid[r][c]}→{output_grid[r][c]}")
        print(f"Changes: {diffs[:5]}")  # Show first 5 changes