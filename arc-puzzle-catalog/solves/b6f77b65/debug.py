import json

with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b6f77b65.json') as f:
    data = json.load(f)

# Look at example 2 (corner=4) more carefully
ex = data['train'][1]
inp = ex['input']
out = ex['output']

print("EXAMPLE 2 (corner=4)")
print("INPUT:")
for row in inp:
    print(row)
print("\nOUTPUT:")
for row in out:
    print(row)

print("\nLooking for rectangular blocks in input...")
# Find rectangles
for color in set(sum(inp, [])):
    if color == 0:
        continue
    positions = []
    for i in range(len(inp)):
        for j in range(len(inp[0])):
            if inp[i][j] == color:
                positions.append((i,j))
    
    if positions:
        min_r = min(p[0] for p in positions)
        max_r = max(p[0] for p in positions)
        min_c = min(p[1] for p in positions)
        max_c = max(p[1] for p in positions)
        print(f"Color {color}: rows {min_r}-{max_r}, cols {min_c}-{max_c} -> {positions}")
        
        # Check where this appears in output
        out_positions = []
        for i in range(len(out)):
            for j in range(len(out[0])):
                if out[i][j] == color:
                    out_positions.append((i,j))
        
        if out_positions:
            out_min_r = min(p[0] for p in out_positions)
            out_max_r = max(p[0] for p in out_positions)
            out_min_c = min(p[1] for p in out_positions)
            out_max_c = max(p[1] for p in out_positions)
            print(f"  -> Output: rows {out_min_r}-{out_max_r}, cols {out_min_c}-{out_max_c}")
            print(f"  -> Shift: ({out_min_r-min_r}, {out_min_c-min_c})")
