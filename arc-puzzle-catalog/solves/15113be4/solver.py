def solve(grid: list[list[int]]) -> list[list[int]]:
    n = len(grid)
    # Find the special color (not 0, 1, or 4)
    color = next(c for row in grid for c in row if c not in (0, 1, 4))

    # Locate the 8x8 box by finding colored cell positions
    cpos = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == color]
    min_r = min(r for r, _ in cpos)
    max_c = max(c for _, c in cpos)
    min_c = min(c for _, c in cpos)

    # Determine which corner the box occupies
    box_r = 0 if min_r < 8 else 15
    box_c = 0 if min_c < 8 else 15

    # Extract the 6x6 interior of the box (skip 4-borders)
    interior = [
        [grid[box_r + 1 + r][box_c + 1 + c] for c in range(6)]
        for r in range(6)
    ]

    # Build 3x3 template from 2x2 blocks in the interior
    template = [
        [
            int(any(
                interior[bi * 2 + dr][bj * 2 + dc] == color
                for dr in range(2) for dc in range(2)
            ))
            for bj in range(3)
        ]
        for bi in range(3)
    ]

    # Identify which grid cells are inside the box
    cell_starts = [0, 4, 8, 12, 16, 20]
    box_crs = {0, 1} if box_r == 0 else {4, 5}
    box_ccs = {0, 1} if box_c == 0 else {4, 5}

    result = [row[:] for row in grid]

    for cr in range(6):
        for cc in range(6):
            if cr in box_crs and cc in box_ccs:
                continue
            rs = cell_starts[cr]
            cs = cell_starts[cc]
            # Check if every template-marked position has a 1
            if all(
                grid[rs + tr][cs + tc] == 1
                for tr in range(3) for tc in range(3)
                if template[tr][tc]
            ):
                for tr in range(3):
                    for tc in range(3):
                        if template[tr][tc]:
                            result[rs + tr][cs + tc] = color

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/15113be4.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
