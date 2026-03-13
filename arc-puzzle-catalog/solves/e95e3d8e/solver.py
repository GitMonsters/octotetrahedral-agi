"""
ARC-AGI Puzzle e95e3d8e Solver

Pattern: The grid is a periodic tiling with rectangular holes filled with 0s.
Solution: Find the minimal tile period, reconstruct the tile from non-zero cells,
and re-tile the entire grid.
"""
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    The grid is a periodic tiling with rectangular holes (0s).
    Find the tile period and reconstruct the full tiling.
    """
    rows = len(grid)
    cols = len(grid[0])

    for rp in range(1, rows + 1):
        for cp in range(1, cols + 1):
            tile = {}
            ok = True
            for r in range(rows):
                if not ok:
                    break
                for c in range(cols):
                    if grid[r][c] != 0:
                        key = (r % rp, c % cp)
                        if key in tile:
                            if tile[key] != grid[r][c]:
                                ok = False
                                break
                        else:
                            tile[key] = grid[r][c]
            if ok and len(tile) == rp * cp:
                return [[tile[(r % rp, c % cp)] for c in range(cols)] for r in range(rows)]
    return grid


if __name__ == "__main__":
    with open("/tmp/arc_task_e95e3d8e.json") as f:
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
            print(f"Test {i}: output generated")
