import json, sys
from collections import Counter

def solve(grid):
    rows, cols = len(grid), len(grid[0])
    g = [row[:] for row in grid]
    
    all_vals = [g[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    by_color = {}
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != bg:
                by_color.setdefault(g[r][c], set()).add((r,c))
    
    for color, comp in by_color.items():
        if len(comp) <= 2:
            continue
        
        # Find best LR axis only (vertical axis of symmetry)
        min_c = min(c for _,c in comp)
        max_c = max(c for _,c in comp)
        best_lr_count = len(comp) + 1
        best_lr_removals = set()
        for axis2 in range(2*min_c, 2*max_c + 1):
            removals = frozenset((r,c) for (r,c) in comp if (r, axis2-c) not in comp)
            if len(removals) < best_lr_count:
                best_lr_count = len(removals)
                best_lr_removals = removals
        
        for (r,c) in best_lr_removals:
            g[r][c] = bg
    
    return g

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        task = json.load(f)

    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        expected = ex['output']
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
