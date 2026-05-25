"""
ARC-AGI puzzle b7cb93ac solver.

Rule: The input contains 3+ colored shapes (connected components by color).
Their total cell count equals H*W for some rectangle. The output tiles all
pieces into that rectangle. The largest piece keeps its original orientation;
smaller pieces may be rotated/flipped. Pieces are tried in color-ascending order.
"""
import json
from typing import List, Optional, Tuple, Dict, Set
from collections import deque


Grid = List[List[int]]
Cells = Tuple[Tuple[int, int], ...]


def normalize(cells: list) -> Cells:
    """Normalize cells to top-left origin, sorted."""
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def get_orientations(cells: Cells) -> List[Cells]:
    """Return all distinct orientations (4 rotations x 2 flips) of a shape."""
    results: List[Cells] = []
    seen: Set[Cells] = set()
    base = list(cells)
    for flip in (False, True):
        if flip:
            max_c = max(c for _, c in base)
            curr = [(r, max_c - c) for r, c in base]
        else:
            curr = list(base)
        for _ in range(4):
            norm = normalize(curr)
            if norm not in seen:
                seen.add(norm)
                results.append(norm)
            max_r = max(r for r, _ in curr)
            curr = [(c, max_r - r) for r, c in curr]
    return results


def extract_components(grid: Grid) -> List[Tuple[int, Cells]]:
    """Extract connected components via flood fill, return (color, normalized_cells)."""
    rows, cols = len(grid), len(grid[0])
    visited: Set[Tuple[int, int]] = set()
    components: List[Tuple[int, Cells]] = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r, c) not in visited:
                color = grid[r][c]
                cells: list = []
                q = deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == color:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                components.append((color, normalize(cells)))
    return components


def solve(grid: Grid) -> Optional[Grid]:
    components = extract_components(grid)
    if not components:
        return grid

    total = sum(len(cells) for _, cells in components)

    # Sort: largest first, then by color ascending for tie-breaking
    components.sort(key=lambda x: (-len(x[1]), x[0]))

    # Precompute orientations per component, original first
    orient_lists: List[List[Cells]] = []
    for _, cells in components:
        orients = get_orientations(cells)
        if cells in orients:
            orients.remove(cells)
            orients.insert(0, cells)
        orient_lists.append(orients)

    # Rectangle dimensions, prefer closer to square
    dims: List[Tuple[int, int]] = []
    for h in range(1, total + 1):
        if total % h == 0:
            dims.append((h, total // h))
    dims.sort(key=lambda d: (abs(d[0] - d[1]), -d[1]))

    def backtrack(out: List[List[int]], remaining: List[int], h: int, w: int) -> bool:
        for r in range(h):
            for c in range(w):
                if out[r][c] == 0:
                    for ii, idx in enumerate(remaining):
                        color = components[idx][0]
                        for orient in orient_lists[idx]:
                            for ar, ac in orient:
                                dr, dc = r - ar, c - ac
                                placed = [(pr + dr, pc + dc) for pr, pc in orient]
                                if all(
                                    0 <= pr < h and 0 <= pc < w and out[pr][pc] == 0
                                    for pr, pc in placed
                                ):
                                    for pr, pc in placed:
                                        out[pr][pc] = color
                                    if backtrack(out, remaining[:ii] + remaining[ii + 1:], h, w):
                                        return True
                                    for pr, pc in placed:
                                        out[pr][pc] = 0
                    return False
        return True

    # Fix largest component in original orientation, try each valid position
    largest_color, largest_cells = components[0]
    bbox_h = max(r for r, _ in largest_cells) + 1
    bbox_w = max(c for _, c in largest_cells) + 1

    # Sort remaining indices by color ascending
    remaining_indices = sorted(range(1, len(components)), key=lambda i: components[i][0])

    for h, w in dims:
        if bbox_h > h or bbox_w > w:
            continue
        for dr in range(h - bbox_h + 1):
            for dc in range(w - bbox_w + 1):
                out = [[0] * w for _ in range(h)]
                for r, c in largest_cells:
                    out[r + dr][c + dc] = largest_color
                if backtrack(out, remaining_indices, h, w):
                    return [row[:] for row in out]

    return None


if __name__ == "__main__":
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/b7cb93ac.json") as f:
        task = json.load(f)

    all_pass = True
    for i, example in enumerate(task["train"]):
        result = solve(example["input"])
        expected = example["output"]
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
            all_pass = False

    for i, example in enumerate(task["test"]):
        result = solve(example["input"])
        print(f"\nTest {i} output: {result}")
        if "output" in example:
            if result == example["output"]:
                print("  PASS")
            else:
                print(f"  FAIL - Expected: {example['output']}")

    if all_pass:
        print("\nAll training examples passed!")
