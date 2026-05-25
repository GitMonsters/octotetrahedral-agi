import copy

def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)

    # Find the two 8s
    eights = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                eights.append((r, c))

    A, B = eights[0], eights[1]
    r1, c1 = A
    r2, c2 = B

    dr = r2 - r1
    dc = c2 - c1
    sign_r = (1 if dr > 0 else -1) if dr != 0 else 0
    sign_c = (1 if dc > 0 else -1) if dc != 0 else 0
    abs_dr = abs(dr)
    abs_dc = abs(dc)

    total_steps = max(abs_dr, abs_dc)
    minor_steps = min(abs_dr, abs_dc)
    major_only = total_steps - minor_steps

    def gen_path(late_minor: bool) -> list[tuple[int, int]]:
        """Generate path from A to B.
        late_minor=True: straight moves first, then diagonal.
        late_minor=False: diagonal moves first, then straight."""
        path = []
        r, c = r1, c1
        for step in range(total_steps):
            if abs_dr >= abs_dc:
                r += sign_r
                if late_minor:
                    if step >= major_only:
                        c += sign_c
                else:
                    if step < minor_steps:
                        c += sign_c
            else:
                c += sign_c
                if late_minor:
                    if step >= major_only:
                        r += sign_r
                else:
                    if step < minor_steps:
                        r += sign_r
            path.append((r, c))
        return path

    path1 = gen_path(late_minor=True)
    path2 = gen_path(late_minor=False)

    for r, c in path1 + path2:
        if (r, c) != A and (r, c) != B:
            result[r][c] = 3

    return result
