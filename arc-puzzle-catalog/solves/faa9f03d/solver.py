import json
import sys
from copy import deepcopy
from collections import deque


def transform(grid):
    R, C = len(grid), len(grid[0])
    result = deepcopy(grid)

    # Step 1: Fill 2-markers iteratively (order: up/down/left/right)
    changed = True
    while changed:
        changed = False
        for r in range(R):
            for c in range(C):
                if result[r][c] == 2:
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < R and 0 <= nc < C and result[nr][nc] not in (0, 2, 4):
                            result[r][c] = result[nr][nc]
                            changed = True
                            break

    # Step 2: Handle 4-markers with extension + corrected abandonment
    # Tracks cells placed by extension so they are pre-locked in gap-fill
    extension_cells = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 4:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and result[nr][nc] not in (0, 2, 4):
                    path_color = result[nr][nc]
                    path_r, path_c = nr, nc
                    ext_dr, ext_dc = -dr, -dc

                    # Extension: fill marker then extend away from path cell to edge
                    result[r][c] = path_color
                    extension_cells.add((r, c))
                    er, ec = r + ext_dr, c + ext_dc
                    while 0 <= er < R and 0 <= ec < C:
                        result[er][ec] = path_color
                        extension_cells.add((er, ec))
                        er += ext_dr
                        ec += ext_dc

                    # Abandonment: walk in path direction, skip at most 1 consecutive zero
                    zeroed = []
                    ar, ac = r + 2 * dr, c + 2 * dc
                    zero_run = 0
                    while 0 <= ar < R and 0 <= ac < C:
                        if result[ar][ac] == path_color:
                            result[ar][ac] = 0
                            zeroed.append((ar, ac))
                            zero_run = 0
                        elif result[ar][ac] == 0:
                            zero_run += 1
                            if zero_run > 1:
                                break
                        else:
                            break
                        ar += dr
                        ac += dc

                    # BFS from zeroed cells to remove connected branches (excluding path/marker)
                    q2 = deque(zeroed)
                    visited = set(zeroed)
                    while q2:
                        br, bc = q2.popleft()
                        for dr2, dc2 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr2, nc2 = br + dr2, bc + dc2
                            if (nr2, nc2) in {(path_r, path_c), (r, c)} | visited:
                                continue
                            if 0 <= nr2 < R and 0 <= nc2 < C and result[nr2][nc2] == path_color:
                                result[nr2][nc2] = 0
                                visited.add((nr2, nc2))
                                q2.append((nr2, nc2))
                    break

    # Step 3: Gap-fill with incremental updates and lock-on-first-change
    # Extension cells are pre-locked (4-marker placed them definitively)
    locked = set(extension_cells)

    def get_component(r0, c0, color, gap_r, gap_c):
        vis = set()
        q = deque([(r0, c0)])
        vis.add((r0, c0))
        while q:
            r2, c2 = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r2 + dr, c2 + dc
                if (nr, nc) == (gap_r, gap_c) or (nr, nc) in vis:
                    continue
                if 0 <= nr < R and 0 <= nc < C and result[nr][nc] == color:
                    vis.add((nr, nc))
                    q.append((nr, nc))
        return vis

    def component_is_L(comp, gap_r, gap_c, axis):
        for r2, c2 in comp:
            if axis == 'H' and r2 != gap_r:
                return True
            if axis == 'V' and c2 != gap_c:
                return True
        return False

    def has_other_neighbor(r2, c2, color, exc_r, exc_c):
        # Extension cells are non-dead-ends (path arm anchored to grid boundary)
        if (r2, c2) in extension_cells:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r2 + dr, c2 + dc
            if (nr, nc) != (exc_r, exc_c) and 0 <= nr < R and 0 <= nc < C and result[nr][nc] == color:
                return True
        return False

    for _ in range(25):
        changed = False
        for r in range(R):
            for c in range(C):
                if (r, c) in locked:
                    continue
                cur = result[r][c]
                if cur in (2, 4):
                    continue

                candidates = []  # (color, is_L_shaped)
                for (r1, c1), (r2, c2), axis in [
                    ((r, c - 1), (r, c + 1), 'H'),
                    ((r - 1, c), (r + 1, c), 'V'),
                ]:
                    if not (0 <= r1 < R and 0 <= c1 < C and 0 <= r2 < R and 0 <= c2 < C):
                        continue
                    v1, v2 = result[r1][c1], result[r2][c2]
                    if v1 != v2 or v1 in (0, 2, 4):
                        continue
                    Y = v1

                    comp1 = get_component(r1, c1, Y, r, c)
                    if (r2, c2) in comp1:
                        continue  # endpoints connected — not a bridging gap

                    # Non-empty cells require both sandwich endpoints to be non-dead-ends
                    if cur != 0:
                        if not (has_other_neighbor(r1, c1, Y, r, c) and
                                has_other_neighbor(r2, c2, Y, r, c)):
                            continue

                    comp2 = get_component(r2, c2, Y, r, c)
                    l_shaped = (component_is_L(comp1, r, c, axis) or
                                component_is_L(comp2, r, c, axis))
                    candidates.append((Y, l_shaped))

                if not candidates:
                    continue

                # L-shaped non-cur candidates win; otherwise min of all candidates
                l_noncur = [Y for Y, l in candidates if l and Y != cur]
                winner = min(l_noncur) if l_noncur else min(Y for Y, _ in candidates)

                if winner != cur:
                    result[r][c] = winner
                    locked.add((r, c))
                    changed = True

        if not changed:
            break

    return result


if __name__ == '__main__':
    task_file = sys.argv[1] if len(sys.argv) > 1 else 'faa9f03d.json'
    with open(task_file) as f:
        task = json.load(f)

    total = 0
    for i, ex in enumerate(task.get('train', [])):
        pred = transform(ex['input'])
        ok = pred == ex['output']
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            for r in range(len(pred)):
                for c in range(len(pred[0])):
                    if pred[r][c] != ex['output'][r][c]:
                        print(f"  ({r},{c}): got {pred[r][c]}, want {ex['output'][r][c]}")
        total += ok

    for i, ex in enumerate(task.get('test', [])):
        pred = transform(ex['input'])
        ok = pred == ex['output']
        print(f"Test  {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            for r in range(len(pred)):
                for c in range(len(pred[0])):
                    if pred[r][c] != ex['output'][r][c]:
                        print(f"  ({r},{c}): got {pred[r][c]}, want {ex['output'][r][c]}")
        total += ok

    print(f"\n{total}/{len(task.get('train',[])) + len(task.get('test',[]))} passing")


solve = transform
