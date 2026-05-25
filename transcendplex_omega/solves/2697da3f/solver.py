def solve(grid: list[list[int]]) -> list[list[int]]:
    # Extract bounding box of non-zero cells
    rows_with = [r for r, row in enumerate(grid) if any(c != 0 for c in row)]
    r0, r1 = rows_with[0], rows_with[-1]
    cols_with = [c for r in range(r0, r1+1) for c, v in enumerate(grid[r]) if v != 0]
    c0, c1 = min(cols_with), max(cols_with)
    
    # Extract shape
    shape = [grid[r][c0:c1+1] for r in range(r0, r1+1)]
    H = len(shape)
    W = len(shape[0])
    
    # Rotation helpers
    def rot90cw(m: list[list[int]]) -> list[list[int]]:
        h, w = len(m), len(m[0])
        return [[m[h-1-j][i] for j in range(h)] for i in range(w)]
    
    def rot180(m: list[list[int]]) -> list[list[int]]:
        h, w = len(m), len(m[0])
        return [[m[h-1-i][w-1-j] for j in range(w)] for i in range(h)]
    
    def rot270cw(m: list[list[int]]) -> list[list[int]]:
        h, w = len(m), len(m[0])
        return [[m[j][w-1-i] for j in range(h)] for i in range(w)]
    
    # Build output grid of size (2W+H) x (2W+H)
    S = 2 * W + H
    out = [[0] * S for _ in range(S)]
    
    def place(block: list[list[int]], row_off: int, col_off: int) -> None:
        for r, row in enumerate(block):
            for c, v in enumerate(row):
                if v != 0:
                    out[row_off + r][col_off + c] = v
    
    # Place 4 rotations in a cross pattern
    place(shape,       W,     0)       # left: original
    place(rot90cw(shape), 0,  W)       # top: 90° CW
    place(rot180(shape), W,   W + H)   # right: 180°
    place(rot270cw(shape), W + H, W)   # bottom: 270° CW
    
    return out

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/2697da3f.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
