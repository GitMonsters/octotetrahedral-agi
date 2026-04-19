"""
ARC-AGI Puzzle 3ed85e70 Solver

Rule: Green (3) walls enclose template patterns (multi-color connected components).
Seeds in the open area are incomplete versions of these templates.
The solver identifies seeds, matches them to templates, and completes them.

Seed types:
1. Single-color filled rectangle matching a template color's bounding box.
   - "Full-size" seed: dimensions = full template → stamp template over it.
   - "Inner" seed: dimensions strictly smaller in BOTH dims → add surrounding color.
   - A color appearing in templates with different color sets is ambiguous → skip.
2. Multi-color fragment: a partial template with unique placement within one template.
"""

import json
import copy
from collections import deque
from typing import Any


def find_connected_components(grid: list[list[int]]) -> list[list[tuple[int, int, int]]]:
    H, W = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    components = []
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != 0 and v != 3 and (r, c) not in visited:
                comp = []
                queue = deque([(r, c)])
                visited.add((r, c))
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc, grid[cr][cc]))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                            nv = grid[nr][nc]
                            if nv != 0 and nv != 3:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                components.append(comp)
    return components


def normalize_component(comp: list[tuple[int, int, int]]):
    min_r = min(r for r, c, v in comp)
    min_c = min(c for r, c, v in comp)
    norm = tuple(sorted((r - min_r, c - min_c, v) for r, c, v in comp))
    return norm, min_r, min_c


def get_pattern_dict(norm: tuple) -> dict[tuple[int, int], int]:
    return {(r, c): v for r, c, v in norm}


def bbox_of_color(norm: tuple, color: int):
    cells = [(r, c) for r, c, v in norm if v == color]
    if not cells:
        return None
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    max_r = max(r for r, c in cells)
    max_c = max(c for r, c in cells)
    return (max_r - min_r + 1, max_c - min_c + 1, min_r, min_c)


def template_dims(norm: tuple) -> tuple[int, int]:
    h = max(r for r, c, v in norm) + 1
    w = max(c for r, c, v in norm) + 1
    return h, w


def solve(input_grid: list[list[int]]) -> list[list[int]]:
    H, W = len(input_grid), len(input_grid[0])
    output = copy.deepcopy(input_grid)

    comps = find_connected_components(input_grid)

    # Classify components into templates (multi-color) and seeds (single-color)
    templates: list[tuple[tuple, list[tuple[int, int]]]] = []
    template_norms: set[tuple] = set()
    single_color_seeds = []

    for comp in comps:
        norm, orig_r, orig_c = normalize_component(comp)
        colors = set(v for r, c, v in norm)

        if len(colors) >= 2:
            if norm not in template_norms:
                template_norms.add(norm)
                templates.append((norm, []))
            for t_norm, origins in templates:
                if t_norm == norm:
                    origins.append((orig_r, orig_c))
                    break
        else:
            color = colors.pop()
            h = max(r for r, c, v in norm) + 1
            w = max(c for r, c, v in norm) + 1
            n_cells = len(norm)
            is_filled = n_cells == h * w
            if is_filled:
                single_color_seeds.append((norm, orig_r, orig_c, color, h, w))

    # Expand single-color seeds
    for s_norm, s_orig_r, s_orig_c, s_color, s_h, s_w in single_color_seeds:
        matches = []
        for t_norm, t_origins in templates:
            t_colors = set(v for r, c, v in t_norm)
            if s_color not in t_colors:
                continue

            bbox = bbox_of_color(t_norm, s_color)
            if bbox is None:
                continue
            bbox_h, bbox_w, bbox_min_r, bbox_min_c = bbox

            if s_h != bbox_h or s_w != bbox_w:
                continue

            t_h, t_w = template_dims(t_norm)
            is_full_size = s_h == t_h and s_w == t_w
            is_strict_inner = s_h < t_h and s_w < t_w

            if not (is_full_size or is_strict_inner):
                continue

            matches.append((t_norm, bbox_min_r, bbox_min_c))

        if len(matches) == 1:
            t_norm, bbox_min_r, bbox_min_c = matches[0]
            t_dict = get_pattern_dict(t_norm)
            t_orig_r = s_orig_r - bbox_min_r
            t_orig_c = s_orig_c - bbox_min_c

            for (dr, dc), v in t_dict.items():
                gr, gc = t_orig_r + dr, t_orig_c + dc
                if 0 <= gr < H and 0 <= gc < W:
                    output[gr][gc] = v

    # Expand multi-color fragments
    for comp in comps:
        norm, orig_r, orig_c = normalize_component(comp)
        colors = set(v for r, c, v in norm)
        if len(colors) < 2:
            continue

        frag_dict = get_pattern_dict(norm)
        frag_h = max(r for r, c, v in norm) + 1
        frag_w = max(c for r, c, v in norm) + 1

        for t_norm, t_origins in templates:
            if norm == t_norm:
                continue

            t_colors = set(v for r, c, v in t_norm)
            if not colors.issubset(t_colors):
                continue

            t_dict = get_pattern_dict(t_norm)
            t_h, t_w = template_dims(t_norm)

            matched_offsets = []
            for off_r in range(-(frag_h - 1), t_h):
                for off_c in range(-(frag_w - 1), t_w):
                    match = True
                    for (fr, fc), fv in frag_dict.items():
                        tr, tc = fr + off_r, fc + off_c
                        if (tr, tc) not in t_dict or t_dict[(tr, tc)] != fv:
                            match = False
                            break
                    if match:
                        matched_offsets.append((off_r, off_c))

            if len(matched_offsets) == 1:
                off_r, off_c = matched_offsets[0]
                t_orig_r = orig_r - off_r
                t_orig_c = orig_c - off_c

                for (dr, dc), v in t_dict.items():
                    gr, gc = t_orig_r + dr, t_orig_c + dc
                    if 0 <= gr < H and 0 <= gc < W:
                        output[gr][gc] = v

    return output


def main():
    import sys
    task_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/arc_task_3ed85e70.json"

    with open(task_file) as f:
        task = json.load(f)

    # Verify on training examples
    all_pass = True
    for ti, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        H, W = len(expected), len(expected[0])
        mismatches = sum(
            1 for r in range(H) for c in range(W) if result[r][c] != expected[r][c]
        )
        status = "PASS" if mismatches == 0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"Train {ti}: {status} ({mismatches} mismatches)")

    # Solve test
    for ti, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"\nTest {ti} solution:")
        print(json.dumps(result))

        # Save solution
        output_path = task_file.replace(".json", f"_solution_{ti}.json")
        with open(output_path, "w") as f:
            json.dump(result, f)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
