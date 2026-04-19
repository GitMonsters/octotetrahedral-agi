import copy

def transform(input_grid):
    H = len(input_grid)
    W = len(input_grid[0])
    output = copy.deepcopy(input_grid)

    # Compute spiral segment lengths: W, H-1, W-1, H-3, W-3, H-5, W-5, ...
    lengths = [W, H - 1, W - 1]
    dec = 3
    while True:
        h_seg = H - dec
        if h_seg <= 0:
            break
        lengths.append(h_seg)
        w_seg = W - dec
        if w_seg <= 0:
            break
        lengths.append(w_seg)
        dec += 2

    # Directions: left, up, right, down
    dirs = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    r, c = H - 1, W - 1  # start at bottom-right corner

    for i, seg_len in enumerate(lengths):
        dr, dc = dirs[i % 4]
        for step in range(seg_len):
            output[r][c] = 9
            if step < seg_len - 1:
                r += dr
                c += dc
        # Turn to next direction and advance one step
        ndr, ndc = dirs[(i + 1) % 4]
        r += ndr
        c += ndc

    return output
