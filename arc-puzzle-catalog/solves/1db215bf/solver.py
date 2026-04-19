def transform(grid):
    H, W = len(grid), len(grid[0])
    N = (H + 1) // 5

    blocks = []
    sep = None
    for i in range(N):
        start = i * 5
        block = [grid[start + r] for r in range(4)]
        blocks.append(block)
        if i < N - 1:
            sep = grid[start + 4][0]

    frames = [b[0][0] for b in blocks]

    # "right" type: inner right col differs from frame, inner left col matches frame
    is_right = []
    for b in blocks:
        frame = b[0][0]
        is_right.append(b[1][2] != frame and b[1][1] == frame)

    # NRC = smallest of {4,6,8} strictly greater than sep (or 8)
    nrc = 8
    for v in [4, 6, 8]:
        if v > sep:
            nrc = v
            break

    # OTHER = max of {4,6,8} without nrc
    others = sorted(set([4, 6, 8]) - {nrc}, reverse=True)
    other = others[0] if others else 8

    first_right = None
    for j in range(N):
        if is_right[j]:
            first_right = j
            break

    out = [None] * N
    for j in range(N):
        if not is_right[j]:
            out[j] = nrc
        elif j == first_right:
            out[j] = other
        else:
            out[j] = sep

    # Right blocks push to j+1
    push_targets = {}
    for j in range(N):
        if is_right[j]:
            frame = frames[j]
            push_val = frame if (frame in {4, 6, 8} and frame != sep) else sep
            target = j + 1
            if target < N and target not in push_targets:
                push_targets[target] = push_val

    # Pushes only override right-block positions
    for target, push_val in push_targets.items():
        if is_right[target]:
            out[target] = push_val

    return [list(out) for _ in range(N)]
