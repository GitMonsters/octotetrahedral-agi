import json, sys

def solve(grid):
    R, C = len(grid), len(grid[0])
    
    # Parse horizontal bands (each band = all same color rows)
    bands = []
    cur_color = grid[0][0]
    cur_count = 0
    for r in range(R):
        c = grid[r][0]
        if c == cur_color:
            cur_count += 1
        else:
            bands.append((cur_color, cur_count))
            cur_color = c
            cur_count = 1
    bands.append((cur_color, cur_count))
    
    # Output size: center = last band thickness x thickness
    # Each preceding band adds 2*thickness to each dimension
    side = 2 * sum(t for _, t in bands) - bands[-1][1]
    
    # Fill from outside in
    out = [[0] * side for _ in range(side)]
    offset = 0
    for color, thickness in bands:
        # Fill the frame at current offset
        inner_side = side - 2 * offset
        for dr in range(inner_side):
            for dc in range(inner_side):
                r, c = offset + dr, offset + dc
                # Check if this cell is within the frame (within thickness of edge)
                if dr < thickness or dr >= inner_side - thickness or \
                   dc < thickness or dc >= inner_side - thickness:
                    out[r][c] = color
        offset += thickness
    
    return out

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        ok = result == ex['output']
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            exp = ex['output']
            for r in range(min(len(result), len(exp))):
                for c in range(min(len(result[0]), len(exp[0]))):
                    if result[r][c] != exp[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, exp {exp[r][c]}")
            if len(result) != len(exp):
                print(f"  Size mismatch: {len(result)}x{len(result[0])} vs {len(exp)}x{len(exp[0])}")
