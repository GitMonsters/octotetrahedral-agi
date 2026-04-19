"""ARC-AGI puzzle 64a7c07e solver.
Rule: Find 8-connected components. Shift each right by its bounding box width.
"""

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Find 8-connected components of non-zero cells
    visited = [[False]*cols for _ in range(rows)]
    components: list[list[tuple[int,int]]] = []
    
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != 0 and not visited[r][c]:
                # BFS for 8-connected component
                comp = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and input_grid[nr][nc] != 0:
                                visited[nr][nc] = True
                                stack.append((nr, nc))
                components.append(comp)
    
    # Build output
    output = [[0]*cols for _ in range(rows)]
    
    for comp in components:
        min_c = min(c for _, c in comp)
        max_c = max(c for _, c in comp)
        width = max_c - min_c + 1
        for r, c in comp:
            output[r][c + width] = input_grid[r][c]
    
    return output


if __name__ == "__main__":
    examples = [
        ([[0,0,0,0,0],[8,8,0,0,0],[8,8,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
         [[0,0,0,0,0],[0,0,8,8,0],[0,0,8,8,0],[0,0,0,0,0],[0,0,0,0,0]]),
        ([[0,0,0,0,0,0,0,0,0,0],[0,8,8,8,0,0,0,0,0,0],[0,8,0,8,0,0,0,0,0,0],[0,8,8,8,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,8,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]],
         [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,8,8,8,0,0,0],[0,0,0,0,8,0,8,0,0,0],[0,0,0,0,8,8,8,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,8,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]),
        ([[0,0,0,0,0,0,0,0,0,0,0,0],[0,8,8,8,8,0,0,0,0,0,0,0],[0,8,8,0,0,0,0,0,0,0,0,0],[0,0,0,8,8,0,0,0,0,0,0,0],[0,8,8,8,8,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,8,8,0,0,0,0,0,0],[0,0,0,0,8,8,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]],
         [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,8,8,8,8,0,0,0],[0,0,0,0,0,8,8,0,0,0,0,0],[0,0,0,0,0,0,0,8,8,0,0,0],[0,0,0,0,0,8,8,8,8,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,8,8,0,0,0,0],[0,0,0,0,0,0,8,8,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]),
    ]
    
    test_input = [[0,0,0,0,0,0,0,0,0,0,0,0],[0,8,0,8,0,8,0,0,0,0,0,0],[0,8,0,8,0,8,0,0,0,0,0,0],[0,0,8,0,8,0,0,0,0,0,0,0],[0,0,8,0,8,0,0,0,0,0,0,0],[0,8,0,8,0,8,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,8,0,0,0,0,0,0],[0,0,0,0,8,8,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
    test_expected = [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,8,0,8,0,8,0],[0,0,0,0,0,0,8,0,8,0,8,0],[0,0,0,0,0,0,0,8,0,8,0,0],[0,0,0,0,0,0,0,8,0,8,0,0],[0,0,0,0,0,0,8,0,8,0,8,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,8,0,0,0,0],[0,0,0,0,0,0,8,8,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]

    all_pass = True
    for i, (inp, exp) in enumerate(examples):
        out = transform(inp)
        if out == exp:
            print(f"Example {i}: PASS")
        else:
            print(f"Example {i}: FAIL")
            for r in range(len(exp)):
                if out[r] != exp[r]:
                    print(f"  Row {r}: got {out[r]}, expected {exp[r]}")
            all_pass = False

    test_out = transform(test_input)
    if test_out == test_expected:
        print("Test: PASS")
    else:
        print("Test: FAIL")
        for r in range(len(test_expected)):
            if test_out[r] != test_expected[r]:
                print(f"  Row {r}: got {test_out[r]}, expected {test_expected[r]}")
        all_pass = False

    print("SOLVED" if all_pass else "FAILED")
