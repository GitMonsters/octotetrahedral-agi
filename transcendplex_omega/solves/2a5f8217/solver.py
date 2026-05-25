def solve(grid: list[list[int]]) -> list[list[int]]:
    """Each shape made of 1s matches a non-1 colored shape by geometry. Replace 1s with matching color."""
    import copy
    rows, cols = len(grid), len(grid[0])
    result = copy.deepcopy(grid)
    visited = [[False]*cols for _ in range(rows)]
    
    def flood_fill(r: int, c: int, color: int) -> list[tuple[int,int]]:
        stack = [(r, c)]
        cells = []
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if visited[cr][cc] or grid[cr][cc] != color:
                continue
            visited[cr][cc] = True
            cells.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc+dc))
        return cells
    
    def normalize(cells: list[tuple[int,int]]) -> frozenset[tuple[int,int]]:
        min_r = min(r for r,c in cells)
        min_c = min(c for r,c in cells)
        return frozenset((r-min_r, c-min_c) for r,c in cells)
    
    # Find all connected components
    one_shapes = []   # list of (cells, normalized_shape)
    color_shapes = {}  # normalized_shape -> color
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0:
                color = grid[r][c]
                cells = flood_fill(r, c, color)
                norm = normalize(cells)
                if color == 1:
                    one_shapes.append((cells, norm))
                else:
                    color_shapes[norm] = color
    
    # Replace each 1-shape with its matching color
    for cells, norm in one_shapes:
        if norm in color_shapes:
            for r, c in cells:
                result[r][c] = color_shapes[norm]
    
    return result


if __name__ == "__main__":
    import json, sys
    with open("/tmp/arc_task_2a5f8217.json") as f:
        task = json.load(f)
    
    all_pass = True
    for i, ex in enumerate(task["train"]):
        out = solve(ex["input"])
        if out == ex["output"]:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            all_pass = False
            for r in range(len(out)):
                if out[r] != ex["output"][r]:
                    print(f"  Row {r}: got {out[r]}")
                    print(f"       exp {ex['output'][r]}")
    
    if all_pass:
        print("\nAll training examples passed!")
    
    # Also produce test output
    for i, ex in enumerate(task["test"]):
        out = solve(ex["input"])
        print(f"\nTest {i} output: {json.dumps(out)}")
        if "output" in ex:
            if out == ex["output"]:
                print(f"Test {i}: PASS")
            else:
                print(f"Test {i}: FAIL")
