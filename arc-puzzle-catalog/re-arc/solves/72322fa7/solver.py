def transform(input_grid: list[list[int]]) -> list[list[int]]:
    grid = [row[:] for row in input_grid]
    rows, cols = len(grid), len(grid[0])

    # Collect non-zero cells
    cells = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]

    # Union-Find clustering with Chebyshev distance <= 2
    parent = {(r, c): (r, c) for r, c, _ in cells}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if max(abs(cells[i][0] - cells[j][0]), abs(cells[i][1] - cells[j][1])) <= 2:
                union((cells[i][0], cells[i][1]), (cells[j][0], cells[j][1]))

    # Group by cluster
    clusters: dict[tuple, list] = {}
    for r, c, v in cells:
        root = find((r, c))
        clusters.setdefault(root, []).append((r, c, v))

    # Separate templates (2 colors) from partials (1 color)
    templates = []
    partials = []
    for cluster in clusters.values():
        colors = set(v for _, _, v in cluster)
        if len(colors) >= 2:
            cc = {}
            for _, _, v in cluster:
                cc[v] = cc.get(v, 0) + 1
            center_color = min(cc, key=cc.get)
            surround_color = [c for c in cc if c != center_color][0]
            center_pos = next((r, c) for r, c, v in cluster if v == center_color)
            offsets = [(r - center_pos[0], c - center_pos[1]) for r, c, v in cluster if v == surround_color]
            templates.append({'cc': center_color, 'sc': surround_color, 'offsets': offsets})
        else:
            partials.append(cluster)

    # Map each color to its template
    c2t = {}
    for t in templates:
        c2t[t['cc']] = t
        c2t[t['sc']] = t

    # Complete each partial
    out = [row[:] for row in grid]
    for partial in partials:
        color = partial[0][2]
        t = c2t.get(color)
        if not t:
            continue
        if color == t['cc']:
            # Partial is center → stamp surround
            for r, c, _ in partial:
                for dr, dc in t['offsets']:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and out[nr][nc] == 0:
                        out[nr][nc] = t['sc']
        else:
            # Partial is surround → find best center, then stamp full pattern
            offsets_set = set(t['offsets'])
            votes: dict[tuple, int] = {}
            for r, c, _ in partial:
                for dr, dc in t['offsets']:
                    ctr = (r - dr, c - dc)
                    votes[ctr] = votes.get(ctr, 0) + 1
            best = max(votes, key=votes.get)
            cr, cc_ = best
            if 0 <= cr < rows and 0 <= cc_ < cols and out[cr][cc_] == 0:
                out[cr][cc_] = t['cc']
            for dr, dc in t['offsets']:
                nr, nc = cr + dr, cc_ + dc
                if 0 <= nr < rows and 0 <= nc < cols and out[nr][nc] == 0:
                    out[nr][nc] = t['sc']
    return out


# === TEST ===
if __name__ == "__main__":
    examples = [
        (
            [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,4,8,4,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,8,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[4,0,4,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,8,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,4,8,4,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,4,8,4],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[4,8,4,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,4,8,4,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
        ),
        (
            [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,8,6,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0],[0,0,3,0,0,0,0,0,0,6,0,0,0,0,0,8,0,8,0,0,0,0],[0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,8,6,8,0,0,0,0,0,0,0,0,0,0,0,8,0,0],[0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,8,6,8,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,1,0,0,0,0,0,8,0,0,0,0,0,0,8,0,0,0,0,0],[0,0,3,0,0,0,0,0,8,6,8,0,0,0,0,8,6,8,0,0,0,0],[0,1,0,1,0,0,0,0,0,8,0,0,0,0,0,0,8,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        ),
        (
            [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,2,8,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,8,0,8,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,2,8,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,8,2,8,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,8,2,8,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]]
        ),
    ]

    all_pass = True
    for i, (inp, exp) in enumerate(examples):
        got = transform(inp)
        if got == exp:
            print(f"Example {i}: PASS")
        else:
            all_pass = False
            print(f"Example {i}: FAIL")
            for r in range(len(exp)):
                for c in range(len(exp[0])):
                    if got[r][c] != exp[r][c]:
                        print(f"  ({r},{c}) got={got[r][c]} exp={exp[r][c]}")

    if all_pass:
        print("\nSOLVED")
    else:
        print("\nFAILED")
