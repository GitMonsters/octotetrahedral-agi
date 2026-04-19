"""ARC puzzle 4a1cacc2: Fill rectangle from dot to nearest corner."""

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    import copy
    grid = copy.deepcopy(input_grid)
    H, W = len(grid), len(grid[0])
    bg = 8
    
    # Find the single non-background pixel
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                color, pr, pc = grid[r][c], r, c
                break
    
    # Find nearest corner by Manhattan distance
    corners = [(0, 0), (0, W-1), (H-1, 0), (H-1, W-1)]
    nearest = min(corners, key=lambda cr: abs(cr[0]-pr) + abs(cr[1]-pc))
    
    # Fill rectangle from pixel to nearest corner
    r_min, r_max = min(pr, nearest[0]), max(pr, nearest[0])
    c_min, c_max = min(pc, nearest[1]), max(pc, nearest[1])
    for r in range(r_min, r_max+1):
        for c in range(c_min, c_max+1):
            grid[r][c] = color
    
    return grid

# --- Verification ---
examples = [
    ([[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,4,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8]],
     [[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[4,4,4,8,8,8,8,8],[4,4,4,8,8,8,8,8],[4,4,4,8,8,8,8,8]]),
    ([[8,8,8,8,8,8],[8,8,8,8,8,8],[8,8,8,8,9,8],[8,8,8,8,8,8],[8,8,8,8,8,8],[8,8,8,8,8,8]],
     [[8,8,8,8,9,9],[8,8,8,8,9,9],[8,8,8,8,9,9],[8,8,8,8,8,8],[8,8,8,8,8,8],[8,8,8,8,8,8]]),
    ([[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,6,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8]],
     [[6,6,6,8,8,8,8,8],[6,6,6,8,8,8,8,8],[6,6,6,8,8,8,8,8],[6,6,6,8,8,8,8,8],[6,6,6,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8]]),
    ([[8,8,6,8],[8,8,8,8],[8,8,8,8],[8,8,8,8]],
     [[8,8,6,6],[8,8,8,8],[8,8,8,8],[8,8,8,8]]),
]

all_pass = True
for i, (inp, exp) in enumerate(examples):
    out = transform(inp)
    ok = out == exp
    print(f"Example {i}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False
        print(f"  Expected: {exp}")
        print(f"  Got:      {out}")

test_input = [[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,4,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8]]
test_expected = [[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,4,4],[8,8,8,8,8,8,8,8,4,4],[8,8,8,8,8,8,8,8,4,4]]
test_out = transform(test_input)
test_ok = test_out == test_expected
print(f"Test: {'PASS' if test_ok else 'FAIL'}")

print("\nSOLVED" if all_pass and test_ok else "\nFAILED")
