def transform(input_grid):
    H = len(input_grid)
    W = len(input_grid[0])
    bg = input_grid[0][0]

    specials = []
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] != bg:
                specials.append((r, c))

    staircase_cells = set()
    for sr, sc in specials:
        # NW direction: left 2, up 2, left 2, up 2, ...
        r, c = sr, sc
        while True:
            count = 0
            for _ in range(2):
                if c - 1 >= 0:
                    c -= 1
                    staircase_cells.add((r, c))
                    count += 1
            if count < 2:
                break
            count = 0
            for _ in range(2):
                if r - 1 >= 0:
                    r -= 1
                    staircase_cells.add((r, c))
                    count += 1
            if count < 2:
                break

        # SE direction: right 2, down 2, right 2, down 2, ...
        r, c = sr, sc
        while True:
            count = 0
            for _ in range(2):
                if c + 1 < W:
                    c += 1
                    staircase_cells.add((r, c))
                    count += 1
            if count < 2:
                break
            count = 0
            for _ in range(2):
                if r + 1 < H:
                    r += 1
                    staircase_cells.add((r, c))
                    count += 1
            if count < 2:
                break

    output = [row[:] for row in input_grid]
    for r, c in staircase_cells:
        if output[r][c] == bg:
            output[r][c] = 3
    return output
