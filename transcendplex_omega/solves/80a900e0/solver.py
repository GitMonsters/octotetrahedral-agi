from collections import defaultdict, deque


def solve(grid):
    R, C = len(grid), len(grid[0])

    bg_even = grid[0][0]
    bg_odd = grid[0][1]

    # Find all non-checkerboard cells
    non_bg = {}
    for r in range(R):
        for c in range(C):
            expected = bg_even if (r + c) % 2 == 0 else bg_odd
            if grid[r][c] != expected:
                non_bg[(r, c)] = grid[r][c]

    if not non_bg:
        return [row[:] for row in grid]

    # Separate into independent diamonds via diagonal connectivity
    visited = set()
    diamonds = []
    for cell in non_bg:
        if cell in visited:
            continue
        component = {}
        queue = deque([cell])
        visited.add(cell)
        while queue:
            r, c = queue.popleft()
            component[(r, c)] = non_bg[(r, c)]
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in non_bg and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        diamonds.append(component)

    output = [row[:] for row in grid]

    def find_lines(cells):
        cell_set = set(cells)
        vis = set()
        lines = []
        for cell in cells:
            if cell in vis:
                continue
            line = []
            q = deque([cell])
            vis.add(cell)
            while q:
                r, c = q.popleft()
                line.append((r, c))
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cell_set and (nr, nc) not in vis:
                        vis.add((nr, nc))
                        q.append((nr, nc))
            lines.append(sorted(line))
        return lines

    for diamond in diamonds:
        # Interior color = most common within this diamond
        color_counts = defaultdict(int)
        for color in diamond.values():
            color_counts[color] += 1
        interior_color = max(color_counts, key=color_counts.get)

        # Edge cells grouped by non-interior color
        color_groups = defaultdict(list)
        for (r, c), color in diamond.items():
            if color != interior_color:
                color_groups[color].append((r, c))

        # Diamond center
        center_r = sum(r for r, c in diamond) / len(diamond)
        center_c = sum(c for r, c in diamond) / len(diamond)

        for color, cells in color_groups.items():
            lines = find_lines(cells)

            for line in lines:
                if len(line) < 2:
                    r0, c0 = line[0]
                    dr = 1 if r0 > center_r else -1
                    dc = 1 if c0 > center_c else -1
                    r, c = r0, c0
                    while True:
                        r += dr
                        c += dc
                        if 0 <= r < R and 0 <= c < C:
                            output[r][c] = color
                        else:
                            break
                    continue

                dr = line[1][0] - line[0][0]
                dc = line[1][1] - line[0][1]
                edge_dir = (1 if dr >= 0 else -1, 1 if dc >= 0 else -1)

                mid_r = sum(r for r, c in line) / len(line)
                mid_c = sum(c for r, c in line) / len(line)

                perp1 = (-edge_dir[1], edge_dir[0])
                perp2 = (edge_dir[1], -edge_dir[0])

                toward_r = mid_r - center_r
                toward_c = mid_c - center_c
                dot1 = perp1[0] * toward_r + perp1[1] * toward_c
                dot2 = perp2[0] * toward_r + perp2[1] * toward_c
                outward = perp1 if dot1 > dot2 else perp2

                for endpoint in [line[0], line[-1]]:
                    r, c = endpoint
                    while True:
                        r += outward[0]
                        c += outward[1]
                        if 0 <= r < R and 0 <= c < C:
                            output[r][c] = color
                        else:
                            break

    return output
