import json, sys

def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    colors = sorted({c for row in grid for c in row})
    c_low, c_high = colors[0], colors[1]
    border_color = 7

    DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def _raw_color_count(r: int, c: int, color: int) -> int:
        """Count cardinal neighbors with the given raw color; off-grid = same."""
        n = 0
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                n += 1
            elif grid[nr][nc] == color:
                n += 1
        return n

    # --- Phase 1: find c_high defects (stray c_high cells in c_low regions) ---
    is_high = [[False] * cols for _ in range(rows)]
    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != c_high or is_high[r][c]:
                    continue
                sc = 0
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid[nr][nc] == c_high and not is_high[nr][nc]:
                            sc += 1
                if sc == 0:
                    is_high[r][c] = True
                    changed = True
                elif sc == 1:
                    # Guard: only erode if every c_low cardinal neighbor is
                    # solidly part of a real c_low region (≥2 c_low neighbours
                    # in the raw grid, counting off-grid as same).  This
                    # prevents cascading when a c_high cell is merely adjacent
                    # to isolated c_low defects.
                    has_opp = False
                    all_solid = True
                    for dr, dc in DIRS:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        if grid[nr][nc] == c_low:
                            has_opp = True
                            if _raw_color_count(nr, nc, c_low) < 2:
                                all_solid = False
                                break
                    if has_opp and all_solid:
                        is_high[r][c] = True
                        changed = True

    # --- Phase 2: build corrected grid (c_high defects → c_low) ---
    corrected = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if is_high[r][c]:
                corrected[r][c] = c_low

    # --- Phase 3: find c_low defects on the corrected grid ---
    is_low = [[False] * cols for _ in range(rows)]
    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                if corrected[r][c] != c_low or is_low[r][c]:
                    continue
                sc = 0
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if corrected[nr][nc] == c_low and not is_low[nr][nc]:
                            sc += 1
                if sc <= 1:
                    is_low[r][c] = True
                    changed = True

    # --- Phase 4: build output ---
    result = [row[:] for row in corrected]
    for r in range(rows):
        for c in range(cols):
            if is_low[r][c]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not is_low[nr][nc]:
                                result[nr][nc] = border_color

    return result

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)
    for i, ex in enumerate(data['train']):
        result = solve(ex['input'])
        print(f"Train {i}: {'PASS' if result == ex['output'] else 'FAIL'}")
