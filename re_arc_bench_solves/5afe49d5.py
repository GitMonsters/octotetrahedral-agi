from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(v for r in grid for v in r).most_common(1)[0][0]

    cc = Counter(v for r in grid for v in r if v != bg)
    if not cc:
        return [[bg] * 3 for _ in range(3)]

    sorted_counts = sorted(cc.values())
    min_count = sorted_counts[0]
    num_colors = len(cc)
    second_min = sorted_counts[1] if len(sorted_counts) > 1 else min_count

    out = [[bg] * 3 for _ in range(3)]

    n = min(min_count, 3)
    for r in range(3 - n, 3):
        out[r][0] = 9

    if min_count > 3 and (second_min - min_count > 1 or num_colors >= 4):
        out[1][1] = 9

    return out
