import json

def compare_grids(inp, out, title):
    print(f"=== {title} ===")
    print(f"Corner: {inp[0][0]}")
    
    print("Input:")
    for i, row in enumerate(inp):
        print(f"{i:2d}: {row}")
    
    print("Output:")
    for i, row in enumerate(out):
        print(f"{i:2d}: {row}")
    
    print()

with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b6f77b65.json') as f:
    data = json.load(f)

# Look at the first few examples carefully
compare_grids(data['train'][1]['input'], data['train'][1]['output'], "Example 2 (corner=4)")
compare_grids(data['train'][2]['input'], data['train'][2]['output'], "Example 3 (corner=6)")

