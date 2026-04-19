def transform(input_grid: list[list[int]]) -> list[list[int]]:
    from collections import Counter

    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Count colors
    cc = Counter()
    for r in range(rows):
        for c in range(cols):
            cc[grid[r][c]] += 1

    colors = list(cc.keys())
    if len(colors) <= 2:
        return grid  # Trivial case: output = input

    bg = cc.most_common(1)[0][0]
    others = [c for c in colors if c != bg]

    # Find connected components for a color
    def find_components(color):
        positions = set()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    positions.add((r, c))
        visited = set()
        components = []
        for p in list(positions):
            if p in visited:
                continue
            comp = set()
            queue = [p]
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                comp.add(cur)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cur[0] + dr, cur[1] + dc
                    if (nr, nc) in positions and (nr, nc) not in visited:
                        queue.append((nr, nc))
            components.append(comp)
        return components

    # Identify donor (2+ components) and recipient (1 component)
    donor_color = None
    recip_color = None
    for c in others:
        comps = find_components(c)
        if len(comps) >= 2:
            donor_color = c
        else:
            recip_color = c

    if donor_color is None or recip_color is None:
        return grid

    donor_comps = find_components(donor_color)

    # Determine recipient's primary wall
    recip_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == recip_color:
                recip_cells.add((r, c))

    wall_counts = {
        'top': sum(1 for r, c in recip_cells if r == 0),
        'bottom': sum(1 for r, c in recip_cells if r == rows - 1),
        'left': sum(1 for r, c in recip_cells if c == 0),
        'right': sum(1 for r, c in recip_cells if c == cols - 1),
    }
    recip_wall = max(wall_counts, key=wall_counts.get)

    # Compute recipient profile (contiguous extent from wall)
    def get_profile(wall):
        if wall == 'bottom':
            prof = []
            for c in range(cols):
                h = 0
                for r in range(rows - 1, -1, -1):
                    if grid[r][c] == recip_color:
                        h = rows - r
                    else:
                        break
                prof.append(h)
            return prof, cols
        elif wall == 'top':
            prof = []
            for c in range(cols):
                h = 0
                for r in range(rows):
                    if grid[r][c] == recip_color:
                        h = r + 1
                    else:
                        break
                prof.append(h)
            return prof, cols
        elif wall == 'left':
            prof = []
            for r in range(rows):
                w = 0
                for c in range(cols):
                    if grid[r][c] == recip_color:
                        w = c + 1
                    else:
                        break
                prof.append(w)
            return prof, rows
        elif wall == 'right':
            prof = []
            for r in range(rows):
                w = 0
                for c in range(cols - 1, -1, -1):
                    if grid[r][c] == recip_color:
                        w = cols - c
                    else:
                        break
                prof.append(w)
            return prof, rows

    profile, n_positions = get_profile(recip_wall)

    # Find defects: positions where 0 < profile < max
    max_prof = max(profile)
    defect_positions = [i for i in range(n_positions) if 0 < profile[i] < max_prof]

    # Group contiguous defects
    defect_groups = []
    if defect_positions:
        current_group = [defect_positions[0]]
        for i in range(1, len(defect_positions)):
            if defect_positions[i] == defect_positions[i - 1] + 1:
                current_group.append(defect_positions[i])
            else:
                defect_groups.append(current_group)
                current_group = [defect_positions[i]]
        defect_groups.append(current_group)

    # For each donor component, compute parallel span and perpendicular profile
    # Parallel = dimension along recipient's wall; perpendicular = away from wall
    donor_info = []
    for comp in donor_comps:
        if recip_wall in ('bottom', 'top'):
            par_positions = sorted(set(c for r, c in comp))
            prof = [sum(1 for r, c in comp if c == p) for p in par_positions]
        else:
            par_positions = sorted(set(r for r, c in comp))
            prof = [sum(1 for r, c in comp if r == p) for p in par_positions]
        donor_info.append({
            'par_positions': par_positions,
            'span_size': len(par_positions),
            'profile': prof,
        })

    # Match defect groups with donor components by span size
    defect_groups_sorted = sorted(defect_groups, key=len)
    donor_info_sorted = sorted(donor_info, key=lambda d: d['span_size'])

    new_profile = profile[:]
    for dg, di in zip(defect_groups_sorted, donor_info_sorted):
        dg_sorted = sorted(dg)
        for j in range(min(len(dg_sorted), len(di['profile']))):
            new_profile[dg_sorted[j]] += di['profile'][j]

    # Reconstruct grid
    output = [[bg] * cols for _ in range(rows)]
    if recip_wall == 'bottom':
        for c in range(cols):
            h = new_profile[c]
            for r in range(rows - h, rows):
                output[r][c] = recip_color
    elif recip_wall == 'top':
        for c in range(cols):
            h = new_profile[c]
            for r in range(h):
                output[r][c] = recip_color
    elif recip_wall == 'left':
        for r in range(rows):
            w = new_profile[r]
            for c in range(w):
                output[r][c] = recip_color
    elif recip_wall == 'right':
        for r in range(rows):
            w = new_profile[r]
            for c in range(cols - w, cols):
                output[r][c] = recip_color

    return output
