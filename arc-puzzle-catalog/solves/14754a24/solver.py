def solve(grid: list[list[int]]) -> list[list[int]]:
    """Find plus-sign shapes made of 4s and 5s, replace the 5s within them with 2s.

    Each cluster of 4s defines part of a plus/cross pattern (center + 4 orthogonal arms).
    All cells of a valid cross must be 4 or 5 (no 0s). The 4s stay; the 5s become 2.
    Edge crosses (where some arms are out of bounds) are allowed.
    """
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    fours = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4:
                fours.add((r, c))

    if not fours:
        return result

    # Find all valid cross (plus-sign) positions
    valid_crosses = []
    for cr in range(rows):
        for cc in range(cols):
            all_arms = [(cr, cc), (cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]
            arms = [(r, c) for r, c in all_arms if 0 <= r < rows and 0 <= c < cols]
            if any(grid[r][c] not in (4, 5) for r, c in arms):
                continue
            cross_fours = frozenset((r, c) for r, c in arms if grid[r][c] == 4)
            if len(cross_fours) >= 2:
                valid_crosses.append((arms, cross_fours))

    valid_crosses.sort(key=lambda x: -len(x[1]))

    # Exact cover via backtracking: assign each 4 to exactly one cross
    def backtrack(remaining, idx, chosen):
        if not remaining:
            return chosen[:]
        if idx >= len(valid_crosses):
            return None
        arms, cross_fours = valid_crosses[idx]
        if cross_fours <= remaining:
            chosen.append(idx)
            res = backtrack(remaining - cross_fours, idx + 1, chosen)
            if res is not None:
                return res
            chosen.pop()
        return backtrack(remaining, idx + 1, chosen)

    solution = backtrack(fours, 0, [])

    if solution:
        for idx in solution:
            arms, _ = valid_crosses[idx]
            for r, c in arms:
                if result[r][c] == 5:
                    result[r][c] = 2

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/14754a24.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
