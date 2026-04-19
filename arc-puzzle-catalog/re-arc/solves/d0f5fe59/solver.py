"""ARC puzzle d0f5fe59: Count objects, output N×N diagonal."""

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(input_grid), len(input_grid[0])
    visited = [[False]*cols for _ in range(rows)]
    
    def flood_fill(r, c):
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if 0 <= cr < rows and 0 <= cc < cols and not visited[cr][cc] and input_grid[cr][cc] == 8:
                visited[cr][cc] = True
                stack.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
    
    n = 0
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] == 8 and not visited[r][c]:
                flood_fill(r, c)
                n += 1
    
    return [[8 if i == j else 0 for j in range(n)] for i in range(n)]


# --- Test against all training examples ---
examples = [
    ([[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,8,0,0,0,0,0,0,0,0],[0,8,8,8,0,0,0,0,0,0,0,0],[0,8,8,0,0,0,0,8,0,0,0,0],[0,0,0,0,0,8,8,8,8,0,0,0],[0,0,0,0,0,8,0,8,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,8,8,0,0,0,0,0,0,0],[0,0,0,8,8,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]],
     [[8,0,0],[0,8,0],[0,0,8]]),
    ([[0,0,0,0,0,0,0,0,0,0],[0,0,8,8,0,0,0,0,0,0],[0,0,8,8,0,0,0,0,0,0],[0,8,8,8,0,0,0,0,0,0],[0,0,8,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,8,8,0,0,0],[0,0,0,0,8,8,8,0,0,0],[0,0,0,0,0,0,8,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,8,8,0,0,0,0,0],[0,8,8,8,0,0,0,0,0,0],[0,0,0,8,0,0,0,8,8,0],[0,0,0,0,0,0,0,8,8,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]],
     [[8,0,0,0],[0,8,0,0],[0,0,8,0],[0,0,0,8]]),
    ([[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,8,8,0,0,0,0,0,0,0,0],[0,0,8,8,8,0,0,0,8,0,0,0],[0,0,0,8,8,0,0,0,8,8,0,0],[0,0,0,0,0,0,0,0,8,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]],
     [[8,0],[0,8]]),
]

test_input = [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,8,0,0],[0,0,0,0,0,8,0,0,8,8,0,0],[0,0,0,8,8,8,0,0,8,8,0,0],[0,0,0,0,8,8,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,8,8,0,0,0,0,0,8,8,0,0],[0,0,8,8,0,0,0,0,8,8,8,0],[0,0,0,8,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,8,8,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
expected = [[8,0,0,0,0],[0,8,0,0,0],[0,0,8,0,0],[0,0,0,8,0],[0,0,0,0,8]]

all_pass = True
for i, (inp, exp) in enumerate(examples):
    out = transform(inp)
    ok = out == exp
    print(f"Example {i}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  Expected: {exp}")
        print(f"  Got:      {out}")
        all_pass = False

test_out = transform(test_input)
test_ok = test_out == expected
print(f"Test: {'PASS' if test_ok else 'FAIL'}")
if not test_ok:
    print(f"  Expected: {expected}")
    print(f"  Got:      {test_out}")

print("\nSOLVED" if all_pass and test_ok else "\nFAILED")
