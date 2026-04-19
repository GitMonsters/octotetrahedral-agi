from collections import deque


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find the enclosed 5-cell (unreachable from border through 5-cells via BFS)
    visited = [[False] * cols for _ in range(rows)]
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and (r == 0 or r == rows - 1 or c == 0 or c == cols - 1):
                visited[r][c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 5:
                visited[nr][nc] = True
                q.append((nr, nc))
    enclosed = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and not visited[r][c]:
                enclosed = (r, c)
                break
        if enclosed:
            break

    # Build height histogram for maximal rectangle detection
    hist = [[0] * cols for _ in range(rows)]
    for c in range(cols):
        for r in range(rows):
            if grid[r][c] == 5:
                hist[r][c] = (hist[r - 1][c] + 1) if r > 0 else 1

    # Find all maximal all-5 rectangles with both dimensions >= 2
    rect_set = set()
    for r in range(rows):
        stack = []
        for c in range(cols + 1):
            h = hist[r][c] if c < cols else 0
            while stack and hist[r][stack[-1]] > h:
                height = hist[r][stack.pop()]
                width = c if not stack else c - stack[-1] - 1
                if height >= 2 and width >= 2:
                    left = stack[-1] + 1 if stack else 0
                    rect_set.add((r - height + 1, left, r, c - 1))
            stack.append(c)

    def border_non5(r1, c1, r2, c2):
        count = 0
        for cc in range(c1, c2 + 1):
            if r1 > 0 and grid[r1 - 1][cc] != 5:
                count += 1
            if r2 < rows - 1 and grid[r2 + 1][cc] != 5:
                count += 1
        for rr in range(r1, r2 + 1):
            if c1 > 0 and grid[rr][c1 - 1] != 5:
                count += 1
            if c2 < cols - 1 and grid[rr][c2 + 1] != 5:
                count += 1
        return count

    # Score each rectangle by area * border_non5_count
    scored = []
    for r1, c1, r2, c2 in rect_set:
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        b = border_non5(r1, c1, r2, c2)
        scored.append((area * b, area, b, r1, c1, r2, c2))
    scored.sort(key=lambda x: (-x[0], -x[1]))

    if not scored:
        return [row[:] for row in grid]

    top_score = scored[0][0]
    tied = [s for s in scored if s[0] == top_score]
    result = [row[:] for row in grid]

    if len(tied) == 1:
        # Unique winner — fill all 5-cells in the rectangle with 9
        _, _, _, r1, c1, r2, c2 = tied[0]
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if result[r][c] == 5:
                    result[r][c] = 9
    else:
        # Tied rectangles — use overlap + enclosed-cell-guided extension selection
        union_cells = set()
        for _, _, _, r1, c1, r2, c2 in tied:
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if grid[r][c] == 5:
                        union_cells.add((r, c))

        rect_a = (tied[0][3], tied[0][4], tied[0][5], tied[0][6])
        rect_b = (tied[1][3], tied[1][4], tied[1][5], tied[1][6]) if len(tied) > 1 else rect_a

        int_r1 = max(rect_a[0], rect_b[0])
        int_c1 = max(rect_a[1], rect_b[1])
        int_r2 = min(rect_a[2], rect_b[2])
        int_c2 = min(rect_a[3], rect_b[3])

        if int_r1 <= int_r2 and int_c1 <= int_c2 and enclosed:
            intersection = set()
            for r in range(int_r1, int_r2 + 1):
                for c in range(int_c1, int_c2 + 1):
                    if grid[r][c] == 5:
                        intersection.add((r, c))

            er, ec = enclosed
            a_has_row = rect_a[0] <= er <= rect_a[2]
            b_has_row = rect_b[0] <= er <= rect_b[2]
            keep_cells = set()

            def add_row_ext(rect, toward_enclosed):
                if toward_enclosed:
                    ext_row = rect[0] if abs(er - rect[0]) <= abs(er - rect[2]) else rect[2]
                else:
                    ext_row = rect[2] if abs(er - rect[0]) <= abs(er - rect[2]) else rect[0]
                for c in range(rect[1], rect[3] + 1):
                    if grid[ext_row][c] == 5 and (ext_row, c) not in intersection:
                        keep_cells.add((ext_row, c))

            def add_col_ext(rect, away_from_enclosed):
                if away_from_enclosed:
                    ext_col = rect[3] if ec < (rect[1] + rect[3]) / 2 else rect[1]
                else:
                    ext_col = rect[1] if ec < (rect[1] + rect[3]) / 2 else rect[3]
                for r in range(rect[0], rect[2] + 1):
                    if grid[r][ext_col] == 5 and (r, ext_col) not in intersection:
                        keep_cells.add((r, ext_col))

            if a_has_row and not b_has_row:
                add_row_ext(rect_a, toward_enclosed=True)
                add_col_ext(rect_b, away_from_enclosed=True)
            elif b_has_row and not a_has_row:
                add_row_ext(rect_b, toward_enclosed=True)
                add_col_ext(rect_a, away_from_enclosed=True)
            else:
                keep_cells = union_cells - intersection

            # Keep intersection cells adjacent to any kept extension cell
            kept_intersection = set()
            for r, c in intersection:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (r + dr, c + dc) in keep_cells:
                        kept_intersection.add((r, c))
                        break

            for r, c in keep_cells | kept_intersection:
                result[r][c] = 9
        else:
            for r, c in union_cells:
                result[r][c] = 9

    return result
