from typing import List, Tuple, Optional
from collections import Counter, deque


def transform(grid: List[List[int]]) -> List[List[int]]:
    R, C = len(grid), len(grid[0])

    counts = Counter(v for row in grid for v in row)
    bg = counts.most_common(1)[0][0]
    non_bg = [c for c in counts if c != bg]

    best_ratio = -1.0
    rect_color: Optional[int] = None
    best_rect: Optional[Tuple[int, int, int, int]] = None

    for color in non_bg:
        total = counts[color]
        area, rect = _largest_solid_rect(grid, R, C, color)
        ratio = area / total
        if ratio > best_ratio:
            best_ratio = ratio
            rect_color = color
            best_rect = rect

    if best_ratio >= 0.5:
        r1, r2, c1, c2 = best_rect  # type: ignore[misc]
        others = [c for c in non_bg if c != rect_color]
        sparse_color = others[0] if others else rect_color
    else:
        sparse_color = non_bg[0]
        cells = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == sparse_color]
        components = _find_components(cells)
        components = [comp for comp in components if len(comp) > 1]
        components = _merge_close_components(components, max_dist=3)
        components.sort(key=len, reverse=True)
        upper_comp, lower_comp = components[0], components[1]

        upper_min_row = min(r for r, _ in upper_comp)
        lower_min_row = min(r for r, _ in lower_comp)
        if upper_min_row > lower_min_row:
            upper_comp, lower_comp = lower_comp, upper_comp

        upper_max_col = max(c for _, c in upper_comp)
        lower_min_col = min(c for _, c in lower_comp)

        c1, c2 = 0, (upper_max_col + lower_min_col) // 2
        r1, r2 = min(r for r, _ in lower_comp), R - 1

    result = [row[:] for row in grid]

    for r in range(R):
        for c in range(C):
            if grid[r][c] != sparse_color:
                continue
            if c1 <= c <= c2:
                if r < r1:
                    for rr in range(r, r1):
                        result[rr][c] = sparse_color
                elif r > r2:
                    for rr in range(r2 + 1, r + 1):
                        result[rr][c] = sparse_color
            elif r1 <= r <= r2:
                if c < c1:
                    for cc in range(c, c1):
                        result[r][cc] = sparse_color
                elif c > c2:
                    for cc in range(c2 + 1, c + 1):
                        result[r][cc] = sparse_color

    return result


def _largest_solid_rect(
    grid: List[List[int]], R: int, C: int, color: int
) -> Tuple[int, Tuple[int, int, int, int]]:
    heights = [0] * C
    best_area = 0
    best_rect = (0, 0, 0, 0)
    for r in range(R):
        for c in range(C):
            heights[c] = heights[c] + 1 if grid[r][c] == color else 0
        area, c1, c2, h = _largest_hist_rect(heights, C)
        if area > best_area:
            best_area = area
            best_rect = (r - h + 1, r, c1, c2)
    return best_area, best_rect


def _largest_hist_rect(heights: List[int], C: int) -> Tuple[int, int, int, int]:
    stack: List[Tuple[int, int]] = []
    best_area = 0
    best = (0, 0, 0)
    for i in range(C + 1):
        h = heights[i] if i < C else 0
        start = i
        while stack and stack[-1][1] > h:
            idx, sh = stack.pop()
            area = sh * (i - idx)
            if area > best_area:
                best_area = area
                best = (idx, i - 1, sh)
            start = idx
        stack.append((start, h))
    c1, c2, hh = best
    return best_area, c1, c2, hh


def _find_components(cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    cell_set = set(cells)
    visited: set = set()
    components = []
    for cell in cells:
        if cell in visited:
            continue
        comp: List[Tuple[int, int]] = []
        queue = deque([cell])
        visited.add(cell)
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in cell_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components


def _merge_close_components(
    components: List[List[Tuple[int, int]]], max_dist: int
) -> List[List[Tuple[int, int]]]:
    n = len(components)
    if n <= 1:
        return components

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            merged = False
            for r1, c1 in components[i]:
                if merged:
                    break
                for r2, c2 in components[j]:
                    if abs(r1 - r2) + abs(c1 - c2) <= max_dist:
                        union(i, j)
                        merged = True
                        break

    merged_map: dict = {}
    for i in range(n):
        root = find(i)
        if root not in merged_map:
            merged_map[root] = []
        merged_map[root].extend(components[i])
    return list(merged_map.values())
