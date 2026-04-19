def transform(grid):
    H = len(grid)
    W = len(grid[0])
    sep_color = grid[4][0]

    blocks = []
    r = 0
    while r + 4 <= H:
        blocks.append([grid[r + i] for i in range(4)])
        r += 5

    N = len(blocks)
    result_row = []

    for block in blocks:
        sep_positions = set()
        for i in range(4):
            for j in range(W):
                if block[i][j] == sep_color:
                    sep_positions.add((i, j))

        valid_2x2 = []
        for i in range(3):
            for j in range(W - 1):
                if all((i + di, j + dj) in sep_positions for di in range(2) for dj in range(2)):
                    valid_2x2.append((i, j))

        if len(valid_2x2) == 1:
            rp, cp = valid_2x2[0]
            result_row.append(rp * W + cp)
        else:
            result_row.append(3)

    return [list(result_row) for _ in range(N)]
