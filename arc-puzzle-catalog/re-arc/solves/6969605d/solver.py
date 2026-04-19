from collections import Counter


def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color (most common)
    counts = Counter()
    for r in range(rows):
        for c in range(cols):
            counts[input_grid[r][c]] += 1
    bg = counts.most_common(1)[0][0]

    # Find sources (non-background pixels)
    sources = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                sources.append((r, c, input_grid[r][c]))

    # For each cell, assign to nearest source by Chebyshev distance
    # Ties broken by Manhattan distance; unresolvable ties → background
    # Even Chebyshev distance → source color; odd → background
    output = [[bg] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            dists = []
            for sr, sc, scolor in sources:
                cheb = max(abs(r - sr), abs(c - sc))
                manh = abs(r - sr) + abs(c - sc)
                dists.append((cheb, manh, scolor))

            dists.sort()
            min_cheb = dists[0][0]

            # All sources at minimum Chebyshev distance
            candidates = [(ch, ma, col) for ch, ma, col in dists if ch == min_cheb]

            min_manh = candidates[0][1]
            best = [b for b in candidates if b[1] == min_manh]

            if len(best) == 1:
                cheb, _, color = best[0]
                if cheb % 2 == 0:
                    output[r][c] = color
            else:
                # Multiple tied sources — color only if all agree
                colors = set(b[2] for b in best)
                if len(colors) == 1:
                    if best[0][0] % 2 == 0:
                        output[r][c] = colors.pop()

    return output
