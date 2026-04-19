import copy
from collections import deque

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    """
    Rule: There's a blue "template" shape with red markers attached.
    Standalone red objects elsewhere act as scaled anchors.
    The blue shape is scaled (by the standalone red's size vs 1x1 template red)
    and stamped at each standalone red's position, preserving the template's
    spatial relationship between blue cells and red markers.
    """
    rows = len(input_grid)
    cols = len(input_grid[0])
    output = copy.deepcopy(input_grid)

    visited = [[False]*cols for _ in range(rows)]

    def bfs(sr, sc):
        comp = []
        q = deque([(sr, sc)])
        visited[sr][sc] = True
        while q:
            r, c = q.popleft()
            comp.append((r, c, input_grid[r][c]))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and input_grid[nr][nc] != 0:
                    visited[nr][nc] = True
                    q.append((nr, nc))
        return comp

    components = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != 0 and not visited[r][c]:
                components.append(bfs(r, c))

    template_comp = None
    standalone_reds = []
    for comp in components:
        if any(v == 1 for _, _, v in comp):
            template_comp = comp
        else:
            standalone_reds.append(comp)

    if template_comp is None or not standalone_reds:
        return output

    template_blues = [(r, c) for r, c, v in template_comp if v == 1]
    template_red_cells = [(r, c) for r, c, v in template_comp if v == 2]

    min_r = min(r for r, c, _ in template_comp)
    min_c = min(c for r, c, _ in template_comp)

    blue_rel = [(r - min_r, c - min_c) for r, c in template_blues]
    red_rel = [(r - min_r, c - min_c) for r, c in template_red_cells]

    # Info for each standalone red object
    standalone_info = []
    for comp in standalone_reds:
        pts = [(r, c) for r, c, _ in comp]
        tl_r, tl_c = min(r for r,c in pts), min(c for r,c in pts)
        br_r = max(r for r,c in pts)
        h = br_r - tl_r + 1
        standalone_info.append((tl_r, tl_c, h))

    num_treds = len(red_rel)

    def stamp(anchor_r, anchor_c, scale):
        ref_r, ref_c = red_rel[0]
        ro = anchor_r - ref_r * scale
        co = anchor_c - ref_c * scale
        for br, bc in blue_rel:
            for dr in range(scale):
                for dc in range(scale):
                    r = br * scale + ro + dr
                    c = bc * scale + co + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        output[r][c] = 1

    if num_treds == 1:
        for tl_r, tl_c, scale in standalone_info:
            stamp(tl_r, tl_c, scale)
    else:
        ref_r, ref_c = red_rel[0]
        expected_diffs = [(r - ref_r, c - ref_c) for r, c in red_rel]

        from collections import defaultdict
        sg = defaultdict(list)
        for i, (tl_r, tl_c, scale) in enumerate(standalone_info):
            sg[scale].append((i, tl_r, tl_c))

        used = set()
        for scale, members in sg.items():
            mdict = {(tr, tc): idx for idx, tr, tc in members}
            for idx, tl_r, tl_c in members:
                if idx in used:
                    continue
                group = [(tl_r, tl_c)]
                used.add(idx)
                ok = True
                for d_r, d_c in expected_diffs[1:]:
                    er, ec = tl_r + d_r * scale, tl_c + d_c * scale
                    if (er, ec) in mdict and mdict[(er, ec)] not in used:
                        group.append((er, ec))
                        used.add(mdict[(er, ec)])
                    else:
                        ok = False
                        break
                if ok and len(group) == num_treds:
                    stamp(tl_r, tl_c, scale)

    return output


# ─── Testing ───
if __name__ == "__main__":
    examples = [
        (
            [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,2,0,0,0,0,0],[0,0,0,0,0,1,1,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,2,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,2,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,2,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,2,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,2,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,2,0,0,0,0,0],[0,0,0,0,0,1,1,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,2,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,2,0,0,0,0,0,0,0,0,0],[0,1,1,1,0,0,0,0,0,2,0,0],[0,0,1,0,0,0,0,0,1,1,1,0],[0,0,2,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,2,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
        ),
        (
            [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,2,1,1,2,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,2,2,0,0,0,0,2,2,0],[0,0,0,2,2,0,0,0,0,2,2,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,2,1,1,2,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,2,2,1,1,1,1,2,2,0],[0,0,0,2,2,1,1,1,1,2,2,0],[0,0,0,0,0,0,0,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0],[0,0,0,0,0,1,1,1,1,1,1,0],[0,0,0,0,0,1,1,1,1,1,1,0]]
        ),
        (
            [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,1,2,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,2,2,2,0,0],[0,0,0,0,0,0,0,2,2,2,0,0],[0,0,0,0,0,0,0,2,2,2,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]],
            [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,1,2,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,2,2,2,0,0],[1,1,1,1,1,1,1,2,2,2,0,0],[1,1,1,1,1,1,1,2,2,2,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
        ),
    ]

    test_input = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0],[0,0,1,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,0],[0,0,0,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    test_expected = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,1,0,2,0,0,0,0,0,0,0,0],[0,0,1,0,2,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,2,2,2,0],[0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,2,2,2,0],[0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,2,2,2,0],[0,1,1,0,0,2,2,0,0,0,0,1,1,1,1,1,1,1,1,1,0],[0,1,1,0,0,2,2,0,0,0,0,1,1,1,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    all_pass = True
    for i, (inp, exp) in enumerate(examples):
        result = transform(inp)
        if result == exp:
            print(f"Example {i}: PASS")
        else:
            print(f"Example {i}: FAIL")
            all_pass = False
            for r in range(len(exp)):
                if result[r] != exp[r]:
                    print(f"  Row {r}: got  {result[r]}")
                    print(f"          want {exp[r]}")

    test_result = transform(test_input)
    if test_result == test_expected:
        print("Test: PASS")
    else:
        print("Test: FAIL")
        all_pass = False
        for r in range(len(test_expected)):
            if test_result[r] != test_expected[r]:
                print(f"  Row {r}: got  {test_result[r]}")
                print(f"          want {test_expected[r]}")

    print("\nSOLVED" if all_pass else "\nFAILED")
