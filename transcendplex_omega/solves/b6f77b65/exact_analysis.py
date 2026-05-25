import json

with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b6f77b65.json') as f:
    data = json.load(f)

# Analyze example 2 (corner=4) pixel by pixel
ex2_input = data['train'][1]['input']
ex2_output = data['train'][1]['output']

print("Example 2 - EXACT pixel analysis")
print("Corner = 4")

# For each row, compare input vs output
for r in range(12):
    inp_row = ex2_input[r]
    out_row = ex2_output[r]
    changes = []
    for c in range(12):
        if inp_row[c] != out_row[c]:
            changes.append(f"{c}:{inp_row[c]}->{out_row[c]}")
    
    if changes:
        print(f"Row {r}: {' '.join(changes)}")
    else:
        print(f"Row {r}: no changes")

print()
print("Looking for the pattern:")
print("Where each input element moved to in output...")

# Track where each non-zero input element went
for r in range(12):
    for c in range(12):
        if ex2_input[r][c] != 0:
            color = ex2_input[r][c]
            # Find where this appeared in output
            found = False
            for out_r in range(12):
                for out_c in range(12):
                    if (ex2_output[out_r][out_c] == color and 
                        ex2_input[out_r][out_c] != color):
                        print(f"Color {color} at ({r},{c}) -> likely moved to ({out_r},{out_c})")
                        found = True
                        break
                if found:
                    break

