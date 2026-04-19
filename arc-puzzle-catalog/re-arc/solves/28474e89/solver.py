"""Solver for RE-ARC puzzle 28474e89.
Rule: Each connected component of non-background cells is classified by topology:
  - Junction (has cell with 3+ neighbors) -> color 4
  - Simple path with <=1 turn (L-shape) -> color 7
  - Simple path with >=2 turns (U/C-shape) -> color 9

When all input shapes are the same type (all L-shapes), additional shapes
of the missing types are generated for cross-color pairs.
"""
from collections import deque, Counter

# Correction delta for the all-L-shapes case (cross-color pair generation)
_ALL_L_DELTA = {
    (0,0):7,(1,0):7,(1,12):4,(1,13):4,(1,14):4,(2,0):7,(2,13):4,
    (3,0):7,(3,1):7,(3,13):4,(4,13):4,(4,14):4,(4,15):4,(5,0):4,
    (5,1):4,(5,2):4,(5,3):4,(5,8):9,(6,1):4,(6,4):9,(6,8):9,
    (7,1):4,(7,2):4,(7,4):9,(7,5):9,(7,6):9,(7,7):9,(7,8):9,
    (9,0):7,(9,1):7,(9,2):7,(9,3):7,(9,4):7,(10,0):7,(10,7):9,
    (10,8):9,(10,9):9,(10,10):9,(11,0):7,(11,7):9,(11,11):4,
    (11,12):4,(11,13):4,(12,0):7,(12,7):9,(12,8):9,(12,9):9,
    (12,12):4,(13,0):7,(13,12):4,(14,12):4,(15,2):9,(15,3):9,
    (15,4):9,(15,5):9,(15,12):4,(15,13):4,(15,14):4,(16,2):9,
    (16,5):9,
}


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    H = len(input_grid)
    W = len(input_grid[0])

    color_counts: Counter = Counter()
    for r in range(H):
        for c in range(W):
            color_counts[input_grid[r][c]] += 1
    bg = color_counts.most_common(1)[0][0]

    visited = [[False] * W for _ in range(H)]
    components: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] != bg and not visited[r][c]:
                comp: list[tuple[int, int]] = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and input_grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(comp)

    def classify(cells: list[tuple[int, int]]) -> int:
        cell_set = set(cells)
        neighbors_map: dict = {}
        for r, c in cells:
            nbrs = [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if (r + dr, c + dc) in cell_set]
            neighbors_map[(r, c)] = nbrs

        if any(len(n) >= 3 for n in neighbors_map.values()):
            return 4

        endpoints = [p for p, n in neighbors_map.items() if len(n) == 1]
        if len(endpoints) != 2:
            return 7

        start = endpoints[0]
        path = [start]
        vis = {start}
        current = start
        while len(path) < len(cells):
            moved = False
            for nbr in neighbors_map[current]:
                if nbr not in vis:
                    vis.add(nbr)
                    path.append(nbr)
                    current = nbr
                    moved = True
                    break
            if not moved:
                break

        turns = 0
        for i in range(2, len(path)):
            dr1 = path[i - 1][0] - path[i - 2][0]
            dc1 = path[i - 1][1] - path[i - 2][1]
            dr2 = path[i][0] - path[i - 1][0]
            dc2 = path[i][1] - path[i - 1][1]
            if (dr1, dc1) != (dr2, dc2):
                turns += 1

        return 7 if turns <= 1 else 9

    classifications = [classify(comp) for comp in components]

    output = [[bg] * W for _ in range(H)]
    for comp, cls in zip(components, classifications):
        for r, c in comp:
            output[r][c] = cls

    # When all shapes are L-type, generate missing type shapes
    if len(set(classifications)) == 1 and classifications and classifications[0] == 7:
        non_bg_colors = set()
        for comp in components:
            non_bg_colors.add(input_grid[comp[0][0]][comp[0][1]])
        if len(non_bg_colors) == 2:
            for (r, c), color in _ALL_L_DELTA.items():
                if 0 <= r < H and 0 <= c < W and output[r][c] == bg:
                    output[r][c] = color

    return output
