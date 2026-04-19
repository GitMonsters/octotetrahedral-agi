import numpy as np
from collections import deque

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(input_grid)
    rows, cols = grid.shape

    # Background = most common color
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Mask of non-background cells
    mask = (grid != bg)

    # Find connected components via BFS
    visited = np.zeros_like(mask, dtype=bool)
    patches = []

    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and not visited[r, c]:
                queue = deque([(r, c)])
                visited[r, c] = True
                component = [(r, c)]
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                            component.append((nr, nc))

                min_r = min(p[0] for p in component)
                max_r = max(p[0] for p in component)
                min_c = min(p[1] for p in component)
                max_c = max(p[1] for p in component)

                patch = grid[min_r:max_r+1, min_c:max_c+1].copy()
                patch[patch == bg] = 0
                patches.append(patch)

    # All patches same size — overlay (non-zero wins)
    h, w = patches[0].shape
    result = np.zeros((h, w), dtype=int)
    for patch in patches:
        nonzero = patch != 0
        result[nonzero] = patch[nonzero]

    return result.tolist()


# === TESTING ===
train = [
    (
        [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1],[1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1],[1,0,0,0,0,1,1,1,1,1,3,3,0,0,1,1],[1,0,2,2,0,1,1,1,1,1,3,3,0,0,1,1],[1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,4,4,4,4,1,1,1,1,1,1,1],[1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1],[1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1],[1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],
        [[4,4,4,4],[3,3,0,0],[3,3,0,0],[0,2,2,0]]
    ),
    (
        [[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,0,0,0,0,8,8,8,8,8,8,8,8,8,8,8],[8,1,1,0,0,8,8,8,8,0,0,3,3,8,8,8],[8,0,0,0,0,8,8,8,8,0,0,0,0,8,8,8],[8,8,8,8,8,8,8,8,8,0,0,0,0,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,0,0,0,0,8,8,8,8,8,8,8,8,8],[8,8,8,0,0,2,0,8,8,8,8,8,8,8,8,8],[8,8,8,0,2,2,0,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,0,0,0,0,8,8,8],[8,8,8,8,8,8,8,8,8,0,0,0,4,8,8,8],[8,8,8,8,8,8,8,8,8,0,0,0,4,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
        [[0,0,3,3],[1,1,2,4],[0,2,2,4]]
    ),
    (
        [[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],[9,9,0,1,0,0,0,9,9,9,9,9,9,9,9,9],[9,9,1,1,0,0,0,9,9,9,9,9,9,9,9,9],[9,9,0,1,1,0,0,9,9,9,9,9,9,9,9,9],[9,9,0,0,0,0,0,9,9,9,9,9,9,9,9,9],[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],[9,9,9,9,9,9,0,0,0,0,0,9,9,9,9,9],[9,9,9,9,9,9,0,0,2,2,0,9,9,9,9,9],[9,9,9,9,9,9,0,0,0,2,0,9,9,9,9,9],[9,9,9,9,9,9,0,0,0,2,0,9,9,9,9,9],[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]],
        [[0,1,0,0,0],[1,1,2,2,0],[0,1,1,2,0],[0,0,0,2,0]]
    ),
]

test_input = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,0,2,0,1,1,1,1,0,0,3,1,1,1,1,1],[1,2,2,0,1,1,1,1,0,0,3,1,1,1,1,1],[1,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1],[1,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1],[1,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1],[1,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1],[1,1,1,4,4,4,1,1,1,1,1,0,0,0,1,1],[1,1,1,0,4,0,1,1,1,1,1,0,0,0,1,1],[1,1,1,0,0,0,1,1,1,1,1,6,0,0,1,1],[1,1,1,1,1,1,1,1,1,1,1,6,6,0,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
expected = [[0,2,3],[2,2,3],[4,4,4],[6,4,0],[6,6,0]]

all_pass = True
for i, (inp, out) in enumerate(train):
    result = transform(inp)
    ok = result == out
    print(f"Example {i}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  Expected: {out}")
        print(f"  Got:      {result}")
        all_pass = False

test_result = transform(test_input)
test_ok = test_result == expected
print(f"Test: {'PASS' if test_ok else 'FAIL'}")
if not test_ok:
    print(f"  Expected: {expected}")
    print(f"  Got:      {test_result}")
    all_pass = False

print("\nSOLVED" if all_pass else "\nFAILED")
