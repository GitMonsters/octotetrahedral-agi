"""ARC-AGI puzzle f823c43c solver.

Pattern: The grid has a repeating tile pattern corrupted by noise (color 6).
Remove noise by finding the smallest consistent tile and re-tiling the grid.
"""
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    noise = 6

    for py in range(1, rows + 1):
        for px in range(1, cols + 1):
            tile = [[None] * px for _ in range(py)]
            consistent = True
            for r in range(rows):
                if not consistent:
                    break
                for c in range(cols):
                    if grid[r][c] != noise:
                        tr, tc = r % py, c % px
                        if tile[tr][tc] is None:
                            tile[tr][tc] = grid[r][c]
                        elif tile[tr][tc] != grid[r][c]:
                            consistent = False
                            break

            if consistent and all(
                tile[tr][tc] is not None for tr in range(py) for tc in range(px)
            ):
                return [[tile[r % py][c % px] for c in range(cols)] for r in range(rows)]

    return grid


if __name__ == "__main__":
    with open("/tmp/arc_task_f823c43c.json") as f:
        task = json.load(f)

    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")

    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        if "output" in ex:
            status = "PASS" if result == ex["output"] else "FAIL"
            print(f"Test {i}: {status}")
        else:
            print(f"Test {i}: solved")
            print(json.dumps(result))
