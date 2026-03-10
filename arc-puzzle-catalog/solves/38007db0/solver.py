import json, sys
from collections import Counter

def solve(grid):
    rows, cols = len(grid), len(grid[0])
    
    # Find border color (the grid divider lines)
    border_color = grid[0][0]
    
    # Find panel boundaries by looking for full border rows/cols
    border_rows = [r for r in range(rows) if all(grid[r][c] == border_color for c in range(cols))]
    border_cols = [c for c in range(cols) if all(grid[r][c] == border_color for r in range(rows))]
    
    # Extract panels: between consecutive border rows/cols
    panel_rows = []
    for i in range(len(border_rows) - 1):
        r1, r2 = border_rows[i], border_rows[i+1]
        if r2 - r1 > 1:
            panel_rows.append((r1, r2))
    
    panel_cols = []
    for i in range(len(border_cols) - 1):
        c1, c2 = border_cols[i], border_cols[i+1]
        if c2 - c1 > 1:
            panel_cols.append((c1, c2))
    
    # Extract each panel's content (including borders)
    def get_panel(pr, pc):
        r1, r2 = panel_rows[pr]
        c1, c2 = panel_cols[pc]
        # Include borders
        return tuple(tuple(grid[r][c] for c in range(c1, c2+1)) for r in range(r1, r2+1))
    
    def get_panel_interior(pr, pc):
        r1, r2 = panel_rows[pr]
        c1, c2 = panel_cols[pc]
        return tuple(tuple(grid[r][c] for c in range(c1+1, c2)) for r in range(r1+1, r2))
    
    # For each row of panels, find the odd one out
    odd_panels = []
    for pr in range(len(panel_rows)):
        interiors = []
        for pc in range(len(panel_cols)):
            interiors.append(get_panel_interior(pr, pc))
        
        # Find the one that's different
        # Count occurrences of each pattern
        pattern_count = Counter(interiors)
        
        if len(pattern_count) == 1:
            # All same, just pick first
            odd_panels.append(get_panel(pr, 0))
        else:
            # Find the unique one
            for pc in range(len(panel_cols)):
                if pattern_count[interiors[pc]] == 1:
                    odd_panels.append(get_panel(pr, pc))
                    break
    
    # Build output by stacking odd panels vertically
    out = []
    for panel in odd_panels:
        for row in panel:
            out.append(list(row))
    
    # Remove duplicate border rows between panels
    result = [out[0]]
    for i in range(1, len(out)):
        if all(out[i][c] == border_color for c in range(len(out[i]))) and \
           all(result[-1][c] == border_color for c in range(len(result[-1]))):
            continue
        result.append(out[i])
    
    return result

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        ok = result == ex['output']
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  Size: {len(result)}x{len(result[0])} vs {len(ex['output'])}x{len(ex['output'][0])}")
            for r in range(min(len(result), len(ex['output']))):
                for c in range(min(len(result[0]), len(ex['output'][0]))):
                    if result[r][c] != ex['output'][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, exp {ex['output'][r][c]}")
