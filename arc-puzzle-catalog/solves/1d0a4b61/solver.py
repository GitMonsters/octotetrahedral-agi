def solve(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])

    # Find horizontal period: smallest p where all non-zero pairs agree
    def find_period_h() -> int:
        for p in range(1, W):
            ok = True
            for r in range(H):
                for c in range(W - p):
                    a, b = grid[r][c], grid[r][c + p]
                    if a != 0 and b != 0 and a != b:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return p
        return W

    # Find vertical period
    def find_period_v() -> int:
        for p in range(1, H):
            ok = True
            for r in range(H - p):
                for c in range(W):
                    a, b = grid[r][c], grid[r + p][c]
                    if a != 0 and b != 0 and a != b:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return p
        return H

    hp = find_period_h()
    vp = find_period_v()

    # Build template tile from non-zero values
    template = [[0] * hp for _ in range(vp)]
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0:
                template[r % vp][c % hp] = grid[r][c]

    # Tile the entire grid
    result = [[template[r % vp][c % hp] for c in range(W)] for r in range(H)]
    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/1d0a4b61.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
