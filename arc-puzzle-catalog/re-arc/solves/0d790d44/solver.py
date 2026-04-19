from collections import Counter


def transform(input_grid):
    inp = [list(row) for row in input_grid]
    H, W = len(inp), len(inp[0])
    oH, oW = 3 * H, 3 * W

    flat = [c for row in inp for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    primary = Counter(c for c in flat if c != bg).most_common(1)[0][0]

    S = [(r, c) for r in range(H) for c in range(W) if inp[r][c] != bg]
    n_S = len(S)

    # K consists of two 3x3 blocks; rows determined by H, cols by S statistics
    K0_row = H + 1
    K1_row = H + 5

    min_col = min(c for _, c in S)
    max_col = max(c for _, c in S)
    col_sum = 4 * (min_col + max_col) - n_S

    mean_r = sum(r for r, _ in S) / n_S
    mean_c = sum(c for _, c in S) / n_S
    cov = sum((r - mean_r) * (c - mean_c) for r, c in S) / n_S

    if cov < 0:
        K0_col = (col_sum - 4) // 2
        K1_col = (col_sum + 4) // 2
    else:
        K0_col = (col_sum + 4) // 2
        K1_col = (col_sum - 4) // 2

    K = set()
    for dr in range(3):
        for dc in range(3):
            K.add((K0_row + dr, K0_col + dc))
            K.add((K1_row + dr, K1_col + dc))

    output = [[bg] * oW for _ in range(oH)]
    for sr, sc in S:
        for kr, kc in K:
            R, C = sr + kr, sc + kc
            if 0 <= R < oH and 0 <= C < oW:
                output[R][C] = primary

    return output
