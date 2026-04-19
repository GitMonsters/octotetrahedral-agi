from collections import Counter

def transform(grid):
    cnt = Counter()
    for row in grid:
        for c in row:
            cnt[c] += 1

    # Background = most common color excluding 2 and 4
    bg_cnt = {k: v for k, v in cnt.items() if k not in (2, 4)}
    bg = max(bg_cnt, key=bg_cnt.get) if bg_cnt else 0

    n2 = cnt.get(2, 0)
    signal_colors = {k for k in cnt if k not in (0, 2, 4)}
    n_sig = len(signal_colors)

    if bg != 0:
        ones = (bg - 3) % 5
    else:
        if n_sig <= 1:
            ones = min(n2, 4)
        elif n_sig == 2:
            ones = 1
        else:  # n_sig >= 3
            ones = 4

    fill_order = [(0,0),(1,0),(2,0),(1,1),(0,1),(2,1),(1,2),(0,2),(2,2)]
    out = [[bg]*3 for _ in range(3)]
    for i in range(min(ones, 9)):
        r, c = fill_order[i]
        out[r][c] = 1

    return out
