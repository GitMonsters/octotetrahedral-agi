from collections import Counter


def transform(grid: list[list[int]]) -> list[list[int]]:
    """The grid is a repeating tile pattern with a blob of a single noise color
    painted over part of it. Recover the tile via majority vote across copies,
    then fix any positions where noise dominated the vote."""
    H = len(grid)
    W = len(grid[0])

    for th in range(2, H):
        for tw in range(2, W):
            # Pass 1: build tile by majority vote across all copies
            tile = [[0] * tw for _ in range(th)]
            for tr in range(th):
                for tc in range(tw):
                    vals = []
                    r = tr
                    while r < H:
                        c = tc
                        while c < W:
                            vals.append(grid[r][c])
                            c += tw
                        r += th
                    tile[tr][tc] = Counter(vals).most_common(1)[0][0]

            # Find mismatches between majority-voted tile and input
            mismatch_colors = Counter()
            for r in range(H):
                for c in range(W):
                    if tile[r % th][c % tw] != grid[r][c]:
                        mismatch_colors[grid[r][c]] += 1

            if not mismatch_colors:
                continue

            total = sum(mismatch_colors.values())
            nc, nc_count = mismatch_colors.most_common(1)[0]

            if nc_count < total * 0.5:
                continue

            # Pass 2: fix positions where noise color won the majority
            for tr in range(th):
                for tc in range(tw):
                    if tile[tr][tc] == nc:
                        non_nc = []
                        r = tr
                        while r < H:
                            c = tc
                            while c < W:
                                if grid[r][c] != nc:
                                    non_nc.append(grid[r][c])
                                c += tw
                            r += th
                        if non_nc:
                            tile[tr][tc] = Counter(non_nc).most_common(1)[0][0]

            # Verify all remaining mismatches are a single noise color
            remaining = set()
            for r in range(H):
                for c in range(W):
                    if tile[r % th][c % tw] != grid[r][c]:
                        remaining.add(grid[r][c])

            if len(remaining) <= 1:
                return [[tile[r % th][c % tw] for c in range(W)] for r in range(H)]

    return [row[:] for row in grid]
