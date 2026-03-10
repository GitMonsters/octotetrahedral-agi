import json, sys
from collections import defaultdict
from itertools import permutations
import math


def solve(grid):
    rows, cols = len(grid), len(grid[0])
    output = [row[:] for row in grid]

    # Group cells by color
    color_cells = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color_cells[grid[r][c]].append((r, c))

    # Identify companion pairs: single-cell color adjacent to a multi-cell color
    companions = {}  # companion_color -> (primary_color, offset_dr, offset_dc)
    for color, cells in color_cells.items():
        if len(cells) == 1:
            r, c = cells[0]
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 0:
                        nb_color = grid[nr][nc]
                        if nb_color != color and len(color_cells[nb_color]) > 1:
                            # offset from primary cell to companion cell
                            companions[color] = (nb_color, r - nr, c - nc)
                            break
                if color in companions:
                    break

    def is_valid_line(p1: tuple, p2: tuple) -> bool:
        dr = p2[0] - p1[0]
        dc = p2[1] - p1[1]
        if dr == 0 and dc == 0:
            return False
        return dr == 0 or dc == 0 or abs(dr) == abs(dc)

    def line_cells(p1: tuple, p2: tuple) -> list:
        dr = p2[0] - p1[0]
        dc = p2[1] - p1[1]
        steps = max(abs(dr), abs(dc))
        if steps == 0:
            return [p1]
        sr = (1 if dr > 0 else -1) if dr != 0 else 0
        sc = (1 if dc > 0 else -1) if dc != 0 else 0
        return [(p1[0] + i * sr, p1[1] + i * sc) for i in range(steps + 1)]

    def find_cycle(vertices: list, adj: dict):
        n = len(vertices)
        if n < 3:
            return None
        # Sort by angle from centroid for non-self-intersecting polygon
        cr = sum(v[0] for v in vertices) / n
        cc = sum(v[1] for v in vertices) / n
        sorted_v = sorted(vertices, key=lambda v: math.atan2(-(v[0] - cr), v[1] - cc))
        if all(sorted_v[(i + 1) % n] in adj[sorted_v[i]] for i in range(n)):
            return sorted_v
        rev = sorted_v[::-1]
        if all(rev[(i + 1) % n] in adj[rev[i]] for i in range(n)):
            return rev
        # Brute force
        if n <= 10:
            first = vertices[0]
            for perm in permutations(vertices[1:]):
                cycle = [first] + list(perm)
                if all(cycle[(i + 1) % n] in adj[cycle[i]] for i in range(n)):
                    return cycle
        return None

    def find_path(vertices: list, adj: dict):
        n = len(vertices)
        if n == 1:
            return vertices[:]
        if n == 2:
            return vertices[:] if vertices[1] in adj[vertices[0]] else None

        def dfs(path, visited):
            if len(path) == n:
                return True
            for nb in adj[path[-1]]:
                if nb not in visited:
                    path.append(nb)
                    visited.add(nb)
                    if dfs(path, visited):
                        return True
                    path.pop()
                    visited.remove(nb)
            return False

        for start in vertices:
            path = [start]
            visited = {start}
            if dfs(path, visited):
                return path
        return None

    # Draw primary shapes (connect vertices with lines)
    for color, cells in color_cells.items():
        if color in companions:
            continue
        if len(cells) <= 1:
            continue

        vertices = cells
        adj = defaultdict(set)
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if is_valid_line(vertices[i], vertices[j]):
                    adj[vertices[i]].add(vertices[j])
                    adj[vertices[j]].add(vertices[i])

        if not any(adj[v] for v in vertices):
            continue

        order = None
        is_cycle = False
        if len(vertices) >= 3:
            order = find_cycle(vertices, adj)
            if order:
                is_cycle = True
        if order is None:
            order = find_path(vertices, adj)
        if order is None:
            continue

        n = len(order)
        num_edges = n if is_cycle else n - 1
        for i in range(num_edges):
            j = (i + 1) % n
            for cell in line_cells(order[i], order[j]):
                output[cell[0]][cell[1]] = color

    # Draw companion shadows
    for comp_color, (prim_color, off_r, off_c) in companions.items():
        prim_cells = [(r, c) for r in range(rows) for c in range(cols) if output[r][c] == prim_color]
        for r, c in prim_cells:
            nr, nc = r + off_r, c + off_c
            if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                output[nr][nc] = comp_color

    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/35ab12c3.json") as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Train {i}: PASS ✓")
        else:
            mismatches = sum(1 for r in range(len(expected)) for c in range(len(expected[0])) if result[r][c] != expected[r][c])
            print(f"Train {i}: FAIL - {mismatches} mismatches")
            # Show first few mismatches
            count = 0
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
                        count += 1
                        if count >= 10:
                            break
                if count >= 10:
                    break
            all_pass = False

    for i, ex in enumerate(data["test"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Test {i}: PASS ✓")
        else:
            mismatches = sum(1 for r in range(len(expected)) for c in range(len(expected[0])) if result[r][c] != expected[r][c])
            print(f"Test {i}: FAIL - {mismatches} mismatches")
            count = 0
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
                        count += 1
                        if count >= 10:
                            break
                if count >= 10:
                    break
            all_pass = False

    if all_pass:
        print("\nAll examples pass! ✓")
