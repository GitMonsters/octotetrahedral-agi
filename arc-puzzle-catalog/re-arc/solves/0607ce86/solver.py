import json
import numpy as np
from collections import Counter


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    """Clean a noisy tiled grid by detecting the tile period and using majority voting."""
    grid = np.array(input_grid)
    H, W = grid.shape

    def find_period(axis: int) -> int:
        dim = H if axis == 0 else W
        other = W if axis == 0 else H

        scores: dict[int, float] = {}
        for p in range(2, dim // 2 + 1):
            matches = 0
            total = 0
            for i in range(dim - p):
                for j in range(other):
                    total += 1
                    if axis == 0:
                        v1, v2 = grid[i, j], grid[i + p, j]
                    else:
                        v1, v2 = grid[j, i], grid[j, i + p]
                    if v1 == v2:
                        matches += 1
            scores[p] = matches / total if total > 0 else 0

        if not scores:
            return dim
        max_score = max(scores.values())
        # Pick smallest period with score close to max
        for p in range(2, dim):
            if scores.get(p, 0) >= max_score - 0.03:
                return p
        return dim

    vp = find_period(0)
    hp = find_period(1)

    # Determine active tile counts
    num_v = H // vp
    num_h = W // hp

    # Majority vote for canonical tile (only within active area)
    tile = np.zeros((vp, hp), dtype=int)
    for tr in range(vp):
        for tc in range(hp):
            values = []
            for r in range(tr, num_v * vp, vp):
                for c in range(tc, num_h * hp, hp):
                    values.append(int(grid[r, c]))
            tile[tr, tc] = Counter(values).most_common(1)[0][0]
    output = np.zeros((H, W), dtype=int)
    for r in range(num_v * vp):
        for c in range(num_h * hp):
            output[r, c] = tile[r % vp, c % hp]

    return output.tolist()


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/0607ce86.json") as f:
        task = json.load(f)

    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = transform(ex["input"])
        expected = ex["output"]
        match = result == expected
        print(f"Example {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            r = np.array(result)
            e = np.array(expected)
            diff = np.argwhere(r != e)
            print(f"  {len(diff)} cells differ")
            for d in diff[:10]:
                print(f"  ({d[0]},{d[1]}): got {r[d[0],d[1]]}, expected {e[d[0],d[1]]}")

    print(f"\n{'SOLVED' if all_pass else 'FAILED'}")
