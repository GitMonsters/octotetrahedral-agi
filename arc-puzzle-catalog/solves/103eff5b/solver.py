import copy


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)

    # Find key cells (non-zero, non-8) and 8 cells
    key_positions = []
    eight_positions = []
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0 and v != 8:
                key_positions.append((r, c))
            elif v == 8:
                eight_positions.append((r, c))

    # Key bounding box
    kr = [r for r, c in key_positions]
    kc = [c for r, c in key_positions]
    k_min_r, k_max_r = min(kr), max(kr)
    k_min_c, k_max_c = min(kc), max(kc)
    key_h = k_max_r - k_min_r + 1
    key_w = k_max_c - k_min_c + 1

    # Extract key grid
    key = []
    for r in range(k_min_r, k_max_r + 1):
        row = []
        for c in range(k_min_c, k_max_c + 1):
            v = grid[r][c]
            row.append(v if v != 8 else 0)
        key.append(row)

    # 8-shape bounding box
    er = [r for r, c in eight_positions]
    ec = [c for r, c in eight_positions]
    e_min_r, e_max_r = min(er), max(er)
    e_min_c, e_max_c = min(ec), max(ec)
    shape_h = e_max_r - e_min_r + 1
    shape_w = e_max_c - e_min_c + 1

    # Rotate key 90° clockwise: new[c][key_h-1-r] = old[r][c]
    # Rotated dimensions: key_w rows x key_h cols
    rotated = [[0] * key_h for _ in range(key_w)]
    for r in range(key_h):
        for c in range(key_w):
            rotated[c][key_h - 1 - r] = key[r][c]

    rot_h = key_w  # rows in rotated key
    rot_w = key_h  # cols in rotated key

    # Block size
    block_h = shape_h // rot_h
    block_w = shape_w // rot_w

    # Replace 8 blocks with colors from rotated key
    for br in range(rot_h):
        for bc in range(rot_w):
            color = rotated[br][bc]
            for dr in range(block_h):
                for dc in range(block_w):
                    r = e_min_r + br * block_h + dr
                    c = e_min_c + bc * block_w + dc
                    if 0 <= r < rows and 0 <= c < cols and result[r][c] == 8:
                        result[r][c] = color if color != 0 else 0

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/103eff5b.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
        if status == "FAIL":
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != ex["output"][r][c]:
                        print(f"  Mismatch at ({r},{c}): got {result[r][c]}, expected {ex['output'][r][c]}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
