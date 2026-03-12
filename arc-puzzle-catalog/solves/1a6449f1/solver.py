from collections import defaultdict


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Extract the interior of the largest rectangle border in the grid.

    The grid contains one or more rectangles drawn with a single non-zero color.
    We find the rectangle with the largest interior area and return its contents.
    """
    rows = len(grid)
    cols = len(grid[0])

    # Find all maximal horizontal segments of same non-zero color (length >= 3)
    seg_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r][c] != 0:
                color = grid[r][c]
                start = c
                while c < cols and grid[r][c] == color:
                    c += 1
                end = c - 1
                if end - start >= 2:
                    seg_groups[(color, start, end)].append(r)
            else:
                c += 1

    best_rect = None
    best_area = 0

    for (color, cs, ce), row_list in seg_groups.items():
        row_list.sort()
        for i in range(len(row_list)):
            for j in range(i + 1, len(row_list)):
                r1, r2 = row_list[i], row_list[j]
                # Verify vertical edges are solid
                valid = True
                for r in range(r1, r2 + 1):
                    if grid[r][cs] != color or grid[r][ce] != color:
                        valid = False
                        break
                if not valid:
                    continue
                ir = r2 - r1 - 1
                ic = ce - cs - 1
                if ir > 0 and ic > 0:
                    area = ir * ic
                    if area > best_area:
                        best_area = area
                        best_rect = (r1, cs, r2, ce)

    if best_rect:
        r1, c1, r2, c2 = best_rect
        return [grid[r][c1 + 1:c2] for r in range(r1 + 1, r2)]
    return grid


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/1a6449f1.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
