from collections import Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Background = most frequent color
    bg = Counter(c for row in grid for c in row).most_common(1)[0][0]

    # Bounding box per non-background color
    bboxes: dict[int, list[int]] = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v == bg:
                continue
            if v not in bboxes:
                bboxes[v] = [r, c, r, c]
            else:
                bb = bboxes[v]
                if r < bb[0]: bb[0] = r
                if c < bb[1]: bb[1] = c
                if r > bb[2]: bb[2] = r
                if c > bb[3]: bb[3] = c

    # Build list of (color, height, width) sorted by area descending
    rects = []
    for color, (r1, c1, r2, c2) in bboxes.items():
        h, w = r2 - r1 + 1, c2 - c1 + 1
        rects.append((h * w, color, h, w))
    rects.sort(reverse=True)

    # Output = largest rect size; paint largest-first, smallest-last (front)
    out_h, out_w = rects[0][2], rects[0][3]
    result = [[0] * out_w for _ in range(out_h)]
    for _, color, h, w in rects:
        for r in range(h):
            for c in range(w):
                result[r][c] = color
    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/20818e16.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
