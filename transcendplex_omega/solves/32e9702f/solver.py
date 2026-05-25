"""
ARC-AGI solver for task 32e9702f

Rule:
- Replace all 0s with 5.
- Shift each contiguous horizontal colored segment left by 1 position,
  clipping at the left grid edge.
"""


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    out = [[5] * cols for _ in range(rows)]

    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r][c] != 0:
                color = grid[r][c]
                start = c
                while c < cols and grid[r][c] == color:
                    c += 1
                # Shift segment left by 1
                new_start = max(0, start - 1)
                new_end = new_start + (c - start)
                if new_start == 0 and start == 0:
                    # Can't shift left; truncate from the right
                    new_end = c - 1
                for nc in range(new_start, min(new_end, cols)):
                    out[r][nc] = color
            else:
                c += 1

    return out


if __name__ == "__main__":
    import json, pathlib

    task_path = pathlib.Path(__file__).resolve().parents[2] / "dataset" / "tasks" / "32e9702f.json"
    with open(task_path) as f:
        task = json.load(f)

    all_pass = True
    for phase in ("train", "test"):
        for i, ex in enumerate(task[phase]):
            result = solve(ex["input"])
            ok = result == ex["output"]
            print(f"{phase}[{i}]: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
                for r, (got, exp) in enumerate(zip(result, ex["output"])):
                    if got != exp:
                        print(f"  row {r}: got {got}")
                        print(f"          exp {exp}")
    raise SystemExit(0 if all_pass else 1)
