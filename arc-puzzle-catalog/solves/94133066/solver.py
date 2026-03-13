def solve(grid):
    rows, cols = len(grid), len(grid[0])
    
    # Find rectangle of 1s
    ones = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    r_min = min(r for r, c in ones)
    r_max = max(r for r, c in ones)
    c_min = min(c for r, c in ones)
    c_max = max(c for r, c in ones)
    
    rect_h = r_max - r_min + 1
    rect_w = c_max - c_min + 1
    rect = [[grid[r_min + r][c_min + c] for c in range(rect_w)] for r in range(rect_h)]
    
    # Marker dots outside rectangle
    outside = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0 and (r < r_min or r > r_max or c < c_min or c > c_max):
                outside[v] = (r, c)
    
    # Marker positions inside rectangle (relative coords)
    inside = {}
    for r in range(rect_h):
        for c in range(rect_w):
            if rect[r][c] in outside:
                inside[rect[r][c]] = (r, c)
    
    markers = list(outside.keys())
    a, b, mk_c = markers[0], markers[1], markers[2]
    ir_a, ic_a = inside[a]; er_a, ec_a = outside[a]
    ir_b, ic_b = inside[b]; er_b, ec_b = outside[b]
    ir_c, ic_c = inside[mk_c]; er_c, ec_c = outside[mk_c]
    
    # Internal/external offsets from anchor marker a
    dr_b, dc_b = ir_b - ir_a, ic_b - ic_a
    dr_c, dc_c = ir_c - ir_a, ic_c - ic_a
    dR_b, dC_b = er_b - er_a, ec_b - ec_a
    dR_c, dC_c = er_c - er_a, ec_c - ec_a
    
    # 2x2 transformation: T * [dr; dc] = [dR; dC]
    det = dr_b * dc_c - dr_c * dc_b
    T11 = (dR_b * dc_c - dR_c * dc_b) // det
    T12 = (dR_c * dr_b - dR_b * dr_c) // det
    T21 = (dC_b * dc_c - dC_c * dc_b) // det
    T22 = (dC_c * dr_b - dC_b * dr_c) // det
    
    # Transform all rectangle points
    points = {}
    for r in range(rect_h):
        for c in range(rect_w):
            dr, dc = r - ir_a, c - ic_a
            nr = er_a + T11 * dr + T12 * dc
            nc = ec_a + T21 * dr + T22 * dc
            points[(nr, nc)] = rect[r][c]
    
    # Build output from bounding box
    min_r = min(p[0] for p in points)
    min_c = min(p[1] for p in points)
    max_r = max(p[0] for p in points)
    max_c = max(p[1] for p in points)
    
    out_h = max_r - min_r + 1
    out_w = max_c - min_c + 1
    output = [[1] * out_w for _ in range(out_h)]
    for (r, c), val in points.items():
        output[r - min_r][c - min_c] = val
    
    return output


if __name__ == "__main__":
    import json
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/94133066.json"))
    
    ok = True
    for i, p in enumerate(task["train"]):
        result = solve(p["input"])
        match = result == p["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'} ({len(result)}x{len(result[0])})")
        if not match:
            ok = False
            print(f"  Expected: {len(p['output'])}x{len(p['output'][0])}")
            for r in range(min(len(result), len(p['output']))):
                if result[r] != p['output'][r]:
                    print(f"  Row {r}: got {result[r]}")
                    print(f"       exp {p['output'][r]}")
    
    for i, p in enumerate(task["test"]):
        result = solve(p["input"])
        if "output" in p:
            match = result == p["output"]
            print(f"Test  {i}: {'PASS' if match else 'FAIL'} ({len(result)}x{len(result[0])})")
            if not match:
                ok = False
                print(f"  Expected: {len(p['output'])}x{len(p['output'][0])}")
                for r in range(min(len(result), len(p['output']))):
                    if result[r] != p['output'][r]:
                        print(f"  Row {r}: got {result[r]}")
                        print(f"       exp {p['output'][r]}")
        else:
            print(f"Test  {i}: {len(result)}x{len(result[0])} (no expected output)")
    
    print(f"\n{'ALL PASSED' if ok else 'SOME FAILED'}")
