def solve(grid: list[list[int]]) -> list[list[int]]:
    """Clean noisy tiled grid by finding tile boundaries and majority-voting the canonical tile."""
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])

    # Count non-zero per row and column
    row_counts = [sum(1 for c in range(cols) if grid[r][c] != 0) for r in range(rows)]
    col_counts = [sum(1 for r in range(rows) if grid[r][c] != 0) for c in range(cols)]

    def find_threshold(counts):
        sorted_vals = sorted(set(counts))
        if len(sorted_vals) <= 1:
            return (sorted_vals[0] + 0.5) if sorted_vals else 0.5
        max_gap = 0
        threshold = sorted_vals[-1] * 0.5
        for i in range(len(sorted_vals) - 1):
            gap = sorted_vals[i + 1] - sorted_vals[i]
            if gap > max_gap:
                max_gap = gap
                threshold = (sorted_vals[i] + sorted_vals[i + 1]) / 2
        return threshold

    thr_r = find_threshold(row_counts)
    thr_c = find_threshold(col_counts)

    # Extract consecutive bands of "tile" rows
    def get_bands(counts, threshold, length):
        bands = []
        i = 0
        while i < length:
            if counts[i] >= threshold:
                start = i
                while i < length and counts[i] >= threshold:
                    i += 1
                bands.append((start, i - 1))
            else:
                i += 1
        return bands

    row_bands = get_bands(row_counts, thr_r, rows)
    col_bands = get_bands(col_counts, thr_c, cols)

    # Determine canonical tile size via mode of band sizes
    rsize_ctr = Counter(b[1] - b[0] + 1 for b in row_bands)
    csize_ctr = Counter(b[1] - b[0] + 1 for b in col_bands)
    tile_h = rsize_ctr.most_common(1)[0][0]
    tile_w = csize_ctr.most_common(1)[0][0]

    # Keep only bands matching canonical size
    row_bands = [b for b in row_bands if b[1] - b[0] + 1 == tile_h]
    col_bands = [b for b in col_bands if b[1] - b[0] + 1 == tile_w]

    # Collect all tile instances
    tiles = []
    for rb in row_bands:
        for cb in col_bands:
            tile = []
            for r in range(rb[0], rb[0] + tile_h):
                row = []
                for c in range(cb[0], cb[0] + tile_w):
                    row.append(grid[r][c])
                tile.append(row)
            tiles.append(tile)

    # Majority vote for canonical tile
    canonical = [[0] * tile_w for _ in range(tile_h)]
    for r in range(tile_h):
        for c in range(tile_w):
            votes = Counter()
            for tile in tiles:
                votes[tile[r][c]] += 1
            canonical[r][c] = votes.most_common(1)[0][0]

    # Build clean output
    output = [[0] * cols for _ in range(rows)]
    for rb in row_bands:
        for cb in col_bands:
            for r in range(tile_h):
                for c in range(tile_w):
                    output[rb[0] + r][cb[0] + c] = canonical[r][c]

    return output


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        expected = ex.get('output')
        if expected:
            status = "PASS" if result == expected else "FAIL"
            print(f"Example {i}: {status}")
            if status == "FAIL":
                for r in range(len(expected)):
                    for c in range(len(expected[0])):
                        if result[r][c] != expected[r][c]:
                            print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
        else:
            print(f"Example {i}: no expected output")
