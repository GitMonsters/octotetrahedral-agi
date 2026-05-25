"""
Solver for ARC-AGI task 62593bfd.

Transformation: colored shapes in a grid are projected to either the top or
bottom edge while keeping their column positions and internal structure.

Rule (discovered via decision-tree analysis on all examples):
  1.  Compute the pixel-weighted centroid row of every shape.
  2.  Each shape is initially classified with a two-level rule:
        • If the shape sits in the upper ≈30 % of the grid (center_r / H ≤ 0.3):
              – very far above the centroid (rel_r ≤ −6.5) → TOP  (outlier stays)
              – otherwise                                   → BOT  (swaps to far edge)
        • If the shape sits in the lower ≈70 %:
              – not extremely far below the centroid (rel_r ≤ 9.0) → TOP  (swaps)
              – extremely far below                                → BOT  (outlier stays)
  3.  Pixel-level collision check at each edge; conflicts are resolved by
      flipping the shape whose centre-of-mass is most offset toward that
      edge (moment_r tiebroken by pixel count).
"""

from collections import Counter


def _find_shapes(grid):
    """Return a list of shape dicts using 8-connected flood fill."""
    H, W = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(H) for c in range(W)).most_common(1)[0][0]
    visited = [[False] * W for _ in range(H)]
    shapes = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                pixels = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    pixels.append((cr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                stack.append((nr, nc))
                min_r = min(p[0] for p in pixels)
                max_r = max(p[0] for p in pixels)
                min_c = min(p[1] for p in pixels)
                max_c = max(p[1] for p in pixels)
                rel = frozenset((pr - min_r, pc - min_c) for pr, pc in pixels)
                shapes.append({
                    "color": color,
                    "pixels": set(pixels),
                    "count": len(pixels),
                    "min_r": min_r, "max_r": max_r,
                    "min_c": min_c, "max_c": max_c,
                    "center_r": sum(p[0] for p in pixels) / len(pixels),
                    "center_c": sum(p[1] for p in pixels) / len(pixels),
                    "height": max_r - min_r + 1,
                    "width": max_c - min_c + 1,
                    "rel": rel,
                })
    return shapes, bg


def _moment_r(s):
    """Vertical offset of COM from bounding-box centre (positive = bottom-heavy)."""
    return s["center_r"] - (s["min_r"] + s["max_r"]) / 2


def _edge_pixels(s, H, top: bool):
    """Pixel set when shape is pushed to the given edge."""
    if top:
        return {(r, s["min_c"] + c) for r, c in s["rel"]}
    offset = H - s["height"]
    return {(r + offset, s["min_c"] + c) for r, c in s["rel"]}


def solve(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])
    shapes, bg = _find_shapes(grid)
    if not shapes:
        return [row[:] for row in grid]

    # pixel-weighted centroid row
    all_px = [(r, c) for s in shapes for r, c in s["pixels"]]
    centroid_r = sum(r for r, _ in all_px) / len(all_px)

    # --- initial classification (depth-2 decision tree) ---
    for s in shapes:
        cr_norm = s["center_r"] / H
        rel_r = s["center_r"] - centroid_r
        if cr_norm <= 0.3:
            s["top"] = rel_r <= -6.5          # outlier near top edge stays
        else:
            s["top"] = rel_r <= 9.0           # most lower shapes swap to top

    # --- collision resolution ---
    def _find_collision(top_flag: bool):
        group = [s for s in shapes if s["top"] == top_flag]
        for i in range(len(group)):
            pi = _edge_pixels(group[i], H, top_flag)
            for j in range(i + 1, len(group)):
                if pi & _edge_pixels(group[j], H, top_flag):
                    return group[i], group[j]
        return None

    for _ in range(len(shapes)):          # iterate until stable
        changed = False
        for edge_top in (True, False):
            coll = _find_collision(edge_top)
            if coll:
                s1, s2 = coll
                m1, m2 = _moment_r(s1), _moment_r(s2)
                # at TOP: flip the most bottom-heavy shape to BOT
                # at BOT: flip the most top-heavy shape to TOP
                if edge_top:
                    flip = s1 if (m1 > m2 or (m1 == m2 and s1["count"] >= s2["count"])) else s2
                else:
                    flip = s1 if (m1 < m2 or (m1 == m2 and s1["count"] <= s2["count"])) else s2
                flip["top"] = not flip["top"]
                changed = True
        if not changed:
            break

    # --- build output ---
    out = [[bg] * W for _ in range(H)]
    for s in shapes:
        for r, c in s["rel"]:
            if s["top"]:
                out[r][s["min_c"] + c] = s["color"]
            else:
                out[r + H - s["height"]][s["min_c"] + c] = s["color"]
    return out


# ── validation ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, os

    task_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "dataset", "tasks", "62593bfd.json"
    )
    with open(task_path) as f:
        task = json.load(f)

    all_pass = True
    for section in ("train", "test"):
        for i, pair in enumerate(task[section]):
            result = solve(pair["input"])
            ok = result == pair["output"]
            tag = "PASS" if ok else "FAIL"
            print(f"{section} pair {i}: {tag}")
            if not ok:
                all_pass = False
                # show first difference
                for r in range(len(pair["output"])):
                    if r < len(result) and result[r] != pair["output"][r]:
                        print(f"  row {r}: got {result[r]}")
                        print(f"  row {r}: exp {pair['output'][r]}")
                        break

    print("---")
    print("ALL PASS" if all_pass else "SOME FAILED")
