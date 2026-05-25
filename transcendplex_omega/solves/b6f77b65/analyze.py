import json

with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b6f77b65.json') as f:
    data = json.load(f)

for i, example in enumerate(data['train']):
    print(f"=== EXAMPLE {i+1} ===")
    inp = example['input']
    out = example['output']
    corner = inp[0][0]
    print(f"Corner color: {corner}")
    
    # Find differences
    diffs = []
    for r in range(len(inp)):
        for c in range(len(inp[0])):
            if inp[r][c] != out[r][c]:
                diffs.append(f"({r},{c}): {inp[r][c]} -> {out[r][c]}")
    
    print(f"Differences ({len(diffs)}):")
    for diff in diffs[:10]:  # First 10
        print(f"  {diff}")
    print()
