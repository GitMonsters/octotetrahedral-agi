import copy
from collections import Counter, defaultdict
from itertools import combinations

def transform(input_grid):
    rows, cols = len(input_grid), len(input_grid[0])
    flat = [v for r in input_grid for v in r]
    bg = Counter(flat).most_common(1)[0][0]
    grid = copy.deepcopy(input_grid)
    
    non_bg = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                non_bg[(r,c)] = grid[r][c]
    
    cells_by_color = defaultdict(list)
    for (r,c), v in non_bg.items():
        cells_by_color[v].append((r,c))
    
    def get_orbit(r0, c0, dr, dc):
        return [(r0+dr,c0+dc),(r0+dc,c0-dr),(r0-dr,c0-dc),(r0-dc,c0+dr)]
    
    # Find all valid orbits
    all_orbits = []
    seen_orbit_keys = set()
    
    for v, cells in cells_by_color.items():
        for (r1,c1), (r2,c2) in combinations(cells, 2):
            centers = set()
            # From 90° CW rotation: A→B
            for rr1,cc1,rr2,cc2 in [(r1,c1,r2,c2),(r2,c2,r1,c1)]:
                nr = rr1 + rr2 + cc2 - cc1
                nc = cc1 + cc2 + rr1 - rr2
                if nr % 2 == 0 and nc % 2 == 0:
                    centers.add((nr // 2, nc // 2))
            # From 180° rotation: midpoint
            if (r1+r2) % 2 == 0 and (c1+c2) % 2 == 0:
                centers.add(((r1+r2)//2, (c1+c2)//2))
            
            for r0, c0 in centers:
                if not (0 <= r0 < rows and 0 <= c0 < cols):
                    continue
                dr, dc = r1 - r0, c1 - c0
                if dr == 0 and dc == 0:
                    dr, dc = r2 - r0, c2 - c0
                if dr == 0 and dc == 0:
                    continue
                orbit = get_orbit(r0, c0, dr, dc)
                orbit_key = tuple(sorted(orbit))
                if orbit_key in seen_orbit_keys:
                    continue
                seen_orbit_keys.add(orbit_key)
                if any(not (0<=r<rows and 0<=c<cols) for r,c in orbit):
                    continue
                colors = [grid[r][c] for r,c in orbit]
                if any(cv != bg and cv != v for cv in colors):
                    continue
                count_v = sum(1 for cv in colors if cv == v)
                if count_v < 2:
                    continue
                center_bonus = 1 if grid[r0][c0] == v else 0
                support = count_v + center_bonus
                all_orbits.append((count_v, support, v, orbit, r0, c0))
    
    all_orbits.sort(key=lambda x: (-x[0], -x[1]))
    
    assigned = set()
    confirmed_centers = set()
    
    for phase_min in [4, 3, 2]:
        for count_v, support, v, orbit, r0, c0 in all_orbits:
            if count_v != phase_min:
                continue
            if phase_min == 2 and (r0, c0) not in confirmed_centers:
                continue
            non_bg_in_orbit = [(r,c) for r,c in orbit if grid[r][c] == v]
            if any((r,c) in assigned for r,c in non_bg_in_orbit):
                continue
            fill_ok = True
            for r, c in orbit:
                if grid[r][c] != bg and grid[r][c] != v:
                    fill_ok = False
                    break
            if not fill_ok:
                continue
            for r, c in non_bg_in_orbit:
                assigned.add((r, c))
            for r, c in orbit:
                if grid[r][c] == bg:
                    grid[r][c] = v
            confirmed_centers.add((r0, c0))
    
    return grid
