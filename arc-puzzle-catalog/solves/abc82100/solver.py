import json
import copy
from collections import deque

def solve_abc82100(grid):
    """Solve abc82100: 8-groups define stamp shapes, chains define color mappings."""
    rows = len(grid)
    cols = len(grid[0])
    output = [[0]*cols for _ in range(rows)]  # Start fresh
    
    # Step 1: Find 8-connected groups of 8-cells
    eight_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                eight_cells.add((r, c))
    
    eight_groups = []
    visited_8 = set()
    for cell in eight_cells:
        if cell in visited_8:
            continue
        group = set()
        queue = deque([cell])
        while queue:
            cr, cc = queue.popleft()
            if (cr, cc) in group:
                continue
            group.add((cr, cc))
            visited_8.add((cr, cc))
            # 8-connectivity (including diagonals)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in eight_cells and (nr, nc) not in group:
                        queue.append((nr, nc))
        eight_groups.append(group)
    
    # Step 2: Find 4-connected components of colored (non-0, non-8) cells
    colored_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in (0, 8):
                colored_cells.add((r, c))
    
    colored_components = []
    visited_c = set()
    for cell in colored_cells:
        if cell in visited_c:
            continue
        comp = set()
        queue = deque([cell])
        while queue:
            cr, cc = queue.popleft()
            if (cr, cc) in comp:
                continue
            comp.add((cr, cc))
            visited_c.add((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in colored_cells and (nr, nc) not in comp:
                    queue.append((nr, nc))
        colored_components.append(comp)
    
    # Step 3: Identify chains = 2-cell components with 2 different colors
    chains = []
    source_cells = []
    for comp in colored_components:
        if len(comp) == 2:
            cells = list(comp)
            c1, c2 = grid[cells[0][0]][cells[0][1]], grid[cells[1][0]][cells[1][1]]
            if c1 != c2:
                chains.append(comp)
                continue
        # Everything else is source cells
        source_cells.extend(comp)
    
    # Step 4: Assign each chain to nearest 8-group
    def centroid(group):
        r_sum = sum(r for r, c in group)
        c_sum = sum(c for r, c in group)
        return (r_sum / len(group), c_sum / len(group))
    
    def dist_sq(p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    
    group_centroids = [centroid(g) for g in eight_groups]
    
    chain_assignments = []  # (chain, group_idx, mapping, reference, offsets)
    chain_cells_all = set()
    
    for chain in chains:
        chain_cent = centroid(chain)
        # Find nearest 8-group
        best_gi = min(range(len(eight_groups)), key=lambda i: dist_sq(chain_cent, group_centroids[i]))
        
        cells = list(chain)
        gcent = group_centroids[best_gi]
        
        # Near cell = closer to group centroid, Far cell = farther
        d0 = dist_sq(cells[0], gcent)
        d1 = dist_sq(cells[1], gcent)
        if d0 <= d1:
            near, far = cells[0], cells[1]
        else:
            near, far = cells[1], cells[0]
        
        source_color = grid[far[0]][far[1]]
        target_color = grid[near[0]][near[1]]
        
        # Reference point: near cell + 1 step toward group centroid
        dr = gcent[0] - near[0]
        dc = gcent[1] - near[1]
        # Normalize to unit step
        if abs(dr) >= abs(dc):
            step_r = 1 if dr > 0 else (-1 if dr < 0 else 0)
            step_c = 0
        else:
            step_r = 0
            step_c = 1 if dc > 0 else (-1 if dc < 0 else 0)
        
        # Handle diagonal case: if dr and dc are equal magnitude, step diagonally?
        # Actually for our cases, the step is always axis-aligned
        if abs(dr) > 0 and abs(dc) > 0 and abs(dr) == abs(dc):
            step_r = 1 if dr > 0 else -1
            step_c = 1 if dc > 0 else -1
        
        ref_r = near[0] + step_r
        ref_c = near[1] + step_c
        reference = (ref_r, ref_c)
        
        # Compute offsets
        offsets = [(r - ref_r, c - ref_c) for r, c in eight_groups[best_gi]]
        
        chain_assignments.append({
            'chain': chain,
            'group_idx': best_gi,
            'source_color': source_color,
            'target_color': target_color,
            'reference': reference,
            'offsets': offsets,
        })
        chain_cells_all.update(chain)
    
    # Build mapping: source_color -> (target_color, offsets)
    color_mappings = {}
    for ca in chain_assignments:
        sc = ca['source_color']
        if sc not in color_mappings:
            color_mappings[sc] = ca
    
    # Step 5: Copy non-modified cells, then apply stamps
    # First, copy all non-8, non-chain, non-source cells
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val == 8:
                output[r][c] = 0  # Remove 8s
            elif (r, c) in chain_cells_all:
                output[r][c] = 0  # Remove chain cells
            elif val != 0 and val in color_mappings:
                output[r][c] = 0  # Will be stamped
            else:
                output[r][c] = val  # Keep unchanged
    
    # Step 6: Stamp source cells
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val == 0 or val == 8 or (r,c) in chain_cells_all:
                continue
            if val in color_mappings:
                ca = color_mappings[val]
                target = ca['target_color']
                for dr, dc in ca['offsets']:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        output[nr][nc] = target
    
    return output

# Load and test
with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/abc82100.json') as f:
    task = json.load(f)

# Test on all training examples
all_correct = True
for ti, ex in enumerate(task['train']):
    inp = ex['input']
    expected = ex['output']
    predicted = solve_abc82100(inp)
    
    rows, cols = len(expected), len(expected[0])
    wrong = 0
    diffs = []
    for r in range(rows):
        for c in range(cols):
            if predicted[r][c] != expected[r][c]:
                wrong += 1
                diffs.append((r, c, predicted[r][c], expected[r][c]))
    
    total = rows * cols
    pct = (total - wrong) / total * 100
    status = "✓" if wrong == 0 else "✗"
    print(f"Train {ti}: {total-wrong}/{total} ({pct:.1f}%) {status}")
    if wrong > 0:
        all_correct = False
        for r, c, p, e in diffs[:20]:
            print(f"  ({r},{c}): predicted {p}, expected {e}")

# Test on test case
test_inp = task['test'][0]['input']
test_expected = task['test'][0]['output']
test_pred = solve_abc82100(test_inp)

rows, cols = len(test_expected), len(test_expected[0])
wrong = 0
diffs = []
for r in range(rows):
    for c in range(cols):
        if test_pred[r][c] != test_expected[r][c]:
            wrong += 1
            diffs.append((r, c, test_pred[r][c], test_expected[r][c]))

total = rows * cols
pct = (total - wrong) / total * 100
status = "✓" if wrong == 0 else "✗"
print(f"\nTest: {total-wrong}/{total} ({pct:.1f}%) {status}")
if wrong > 0:
    all_correct = False
    for r, c, p, e in diffs[:30]:
        print(f"  ({r},{c}): predicted {p}, expected {e}")

if all_correct:
    print("\n🎉 ALL EXAMPLES CORRECT!")

solve = solve_abc82100
