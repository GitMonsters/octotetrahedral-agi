def solve(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])
    result = [[0]*W for _ in range(H)]

    # Find the 3x3 non-zero block
    br, bc = -1, -1
    for r in range(H-2):
        for c in range(W-2):
            if grid[r][c] != 0:
                br, bc = r, c
                break
        if br >= 0:
            break

    # Extract the 3x3 block
    block = [[grid[br+dr][bc+dc] for dc in range(3)] for dr in range(3)]

    # Center of block
    cr, cc = br+1, bc+1
    center_val = block[1][1]

    # Copy original block
    for dr in range(3):
        for dc in range(3):
            result[br+dr][bc+dc] = block[dr][dc]

    # Middle row extends horizontally
    left_val = block[1][0]
    right_val = block[1][2]
    for c in range(cc):
        result[cr][c] = left_val
    for c in range(cc+1, W):
        result[cr][c] = right_val

    # Center column extends vertically
    top_val = block[0][1]
    bottom_val = block[2][1]
    for r in range(cr):
        result[r][cc] = top_val
    for r in range(cr+1, H):
        result[r][cc] = bottom_val

    # Four corner diagonals
    tl_val = block[0][0]  # top-left corner
    tr_val = block[0][2]  # top-right corner
    bl_val = block[2][0]  # bottom-left corner
    bri_val = block[2][2] # bottom-right corner

    # Top-left diagonal (up-left from top-left of block)
    r, c = br-1, bc-1
    while r >= 0 and c >= 0:
        result[r][c] = tl_val
        r -= 1; c -= 1

    # Top-right diagonal (up-right from top-right of block)
    r, c = br-1, bc+2+0
    while r >= 0 and c < W:
        result[r][c] = tr_val
        r -= 1; c += 1

    # Bottom-left diagonal (down-left from bottom-left of block)
    r, c = br+3, bc-1
    while r < H and c >= 0:
        result[r][c] = bl_val
        r += 1; c -= 1

    # Bottom-right diagonal (down-right from bottom-right of block)
    r, c = br+3, bc+3
    while r < H and c < W:
        result[r][c] = bri_val
        r += 1; c += 1

    # Restore center
    result[cr][cc] = center_val

    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/1d398264.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
        if status == "FAIL":
            for r in range(len(result)):
                if result[r] != ex["output"][r]:
                    print(f"  Row {r}: got {result[r]}")
                    print(f"       exp {ex['output'][r]}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
