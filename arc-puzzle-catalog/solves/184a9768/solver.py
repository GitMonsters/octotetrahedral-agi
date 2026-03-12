from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find connected components of each non-zero, non-5 color
    visited = [[False] * cols for _ in range(rows)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 5 and not visited[r][c]:
                color = grid[r][c]
                queue = deque([(r, c)])
                visited[r][c] = True
                cells = set()
                while queue:
                    cr, cc = queue.popleft()
                    cells.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append((color, cells))

    # Classify as frame or patch
    frames = []
    patches = []

    for color, cells in components:
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        has_holes = False
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in cells:
                    has_holes = True
                    break
            if has_holes:
                break

        if has_holes:
            frames.append((color, cells, min_r, max_r, min_c, max_c))
        else:
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            patches.append((color, h, w))

    # Copy frames to result
    for color, cells, min_r, max_r, min_c, max_c in frames:
        for r, c in cells:
            result[r][c] = color

    # Build hole sets for each frame
    frame_holes = {}
    for fi, (color, cells, min_r, max_r, min_c, max_c) in enumerate(frames):
        holes = set()
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in cells:
                    holes.add((r, c))
        frame_holes[fi] = holes

    # Precompute valid placements for each patch
    def get_placements(pi, fholes):
        color, h, w = patches[pi]
        placements = []
        for fi in fholes:
            fcolor, fcells, min_r, max_r, min_c, max_c = frames[fi]
            for r in range(min_r, max_r - h + 2):
                for c in range(min_c, max_c - w + 2):
                    if all((r + dr, c + dc) in fholes[fi] for dr in range(h) for dc in range(w)):
                        placements.append((fi, r, c))
        return placements

    # Backtracking solver
    def backtrack(idx, fholes):
        if idx == len(patches):
            return True
        color, h, w = patches[idx]
        placements = get_placements(idx, fholes)
        for fi, r, c in placements:
            cells_to_fill = [(r + dr, c + dc) for dr in range(h) for dc in range(w)]
            for cell in cells_to_fill:
                fholes[fi].remove(cell)
            if backtrack(idx + 1, fholes):
                for cell in cells_to_fill:
                    result[cell[0]][cell[1]] = color
                return True
            for cell in cells_to_fill:
                fholes[fi].add(cell)
        return False

    # Sort patches by number of valid placements (fewest first) for faster pruning
    order = sorted(range(len(patches)), key=lambda i: len(get_placements(i, frame_holes)))
    patches_sorted = [patches[i] for i in order]
    patches = patches_sorted

    backtrack(0, frame_holes)

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/184a9768.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
        if status == "FAIL":
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != ex["output"][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {ex['output'][r][c]}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
