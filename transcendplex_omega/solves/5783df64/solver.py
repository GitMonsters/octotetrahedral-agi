def solve(grid):
    """Divide grid into 3x3 blocks, extract the single non-zero value from each."""
    R, C = len(grid), len(grid[0])
    bh, bw = R // 3, C // 3
    out = []
    for bi in range(3):
        row = []
        for bj in range(3):
            val = 0
            for r in range(bi * bh, (bi + 1) * bh):
                for c in range(bj * bw, (bj + 1) * bw):
                    if grid[r][c] != 0:
                        val = grid[r][c]
            row.append(val)
        out.append(row)
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
