#!/usr/bin/env python3
"""
Solver for ARC-AGI task 40f6cd08.

Pattern: A template rectangle has a nested color structure.
Single-color target rectangles are filled with a scaled version of the template's
inner structure, preserving the outer border.
"""

import json
import sys
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    """Solve the puzzle."""
    
    result = [row[:] for row in grid]
    
    # Find all rectangles
    rects = find_rectangles(grid)
    
    if not rects:
        return result
    
    # The most complex rectangle is the template
    rects.sort(key=lambda r: r['num_colors'], reverse=True)
    template_rect = rects[0]
    
    # Find target rectangles (single-color, not the template)
    # Note: targets can have the same color as the template border
    template_bounds = set(template_rect['bounds'])
    target_rects = [
        r for r in rects[1:] 
        if r['num_colors'] == 1 and tuple(r['bounds']) != template_bounds
    ]
    
    if not target_rects:
        return result
    
    # Extract template pattern
    tr1, tr2, tc1, tc2 = template_rect['bounds']
    template_pattern = [
        [grid[i][j] for j in range(tc1, tc2 + 1)]
        for i in range(tr1, tr2 + 1)
    ]
    
    # Find border color and width (most common color = border)
    border_color = template_rect['color']
    border_w = find_border_width(template_pattern, border_color)
    
    # Extract inner pattern
    template_h = len(template_pattern)
    template_w = len(template_pattern[0])
    template_inner = [
        template_pattern[i][border_w:template_w - border_w]
        for i in range(border_w, template_h - border_w)
    ]
    
    # Apply to each target
    for target_rect in target_rects:
        target_r1, target_r2, target_c1, target_c2 = target_rect['bounds']
        target_h = target_r2 - target_r1 + 1
        target_w = target_c2 - target_c1 + 1
        
        # Calculate inner size for target (same border as template)
        inner_h = target_h - 2 * border_w
        inner_w = target_w - 2 * border_w
        
        if inner_h > 0 and inner_w > 0:
            # Scale template's inner to target's inner size
            scaled_inner = scale_nearest_neighbor(template_inner, inner_h, inner_w)
            
            # Fill target: keep border, fill inner with scaled
            for i in range(target_h):
                for j in range(target_w):
                    if i < border_w or i >= target_h - border_w or \
                       j < border_w or j >= target_w - border_w:
                        # Border - keep original value (should be target's color = all same in input)
                        result[target_r1 + i][target_c1 + j] = grid[target_r1 + i][target_c1 + j]
                    else:
                        # Inner - use scaled pattern
                        inner_i = i - border_w
                        inner_j = j - border_w
                        result[target_r1 + i][target_c1 + j] = scaled_inner[inner_i][inner_j]
    
    return result


def find_rectangles(grid: List[List[int]]):
    """Find all connected rectangular regions."""
    visited = set()
    rects = []
    
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != 0 and (i, j) not in visited:
                color = grid[i][j]
                cells = set()
                q = [(i, j)]
                visited.add((i, j))
                
                # BFS
                while q:
                    r, c = q.pop(0)
                    cells.add((r, c))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and 
                            (nr, nc) not in visited and grid[nr][nc] == color):
                            visited.add((nr, nc))
                            q.append((nr, nc))
                
                # Bounds
                min_r = min(c[0] for c in cells)
                max_r = max(c[0] for c in cells)
                min_c = min(c[1] for c in cells)
                max_c = max(c[1] for c in cells)
                
                # Count distinct colors in bounding box
                colors = set()
                for r in range(min_r, max_r + 1):
                    for c in range(min_c, max_c + 1):
                        if grid[r][c] != 0:
                            colors.add(grid[r][c])
                
                rects.append({
                    'color': color,
                    'bounds': (min_r, max_r, min_c, max_c),
                    'num_colors': len(colors),
                })
    
    return rects


def find_border_width(pattern: List[List[int]], border_color: int) -> int:
    """Find how many layers of border_color surround the pattern."""
    if not pattern:
        return 0
    
    h = len(pattern)
    w = len(pattern[0])
    
    # Find when border_color stops appearing at left edge
    left_border = 0
    for j in range(w // 2):
        if all(pattern[i][j] == border_color for i in range(h)):
            left_border = j + 1
        else:
            break
    
    # Find when border_color stops appearing at top edge
    top_border = 0
    for i in range(h // 2):
        if all(pattern[i][j] == border_color for j in range(w)):
            top_border = i + 1
        else:
            break
    
    return min(left_border, top_border)


def scale_nearest_neighbor(data: List[List[int]], target_h: int, target_w: int) -> List[List[int]]:
    """Scale using nearest-neighbor."""
    if not data or target_h == 0 or target_w == 0:
        return [[0] * target_w for _ in range(target_h)]
    
    src_h = len(data)
    src_w = len(data[0])
    
    result = []
    for i in range(target_h):
        row = []
        for j in range(target_w):
            src_i = min(int(i * src_h / target_h), src_h - 1)
            src_j = min(int(j * src_w / target_w), src_w - 1)
            row.append(data[src_i][src_j])
        result.append(row)
    
    return result


if __name__ == '__main__':
    if len(sys.argv) > 1:
        task_path = sys.argv[1]
    else:
        task_path = '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/40f6cd08.json'
    
    with open(task_path) as f:
        task = json.load(f)
    
    all_pass = True
    for i, example in enumerate(task['train']):
        output = solve(example['input'])
        expected = example['output']
        
        if output == expected:
            print(f"Training example {i}: PASS")
        else:
            print(f"Training example {i}: FAIL")
            all_pass = False
            # Find first difference
            for ri in range(len(output)):
                for ci in range(len(output[0])):
                    if output[ri][ci] != expected[ri][ci]:
                        print(f"  Diff at ({ri}, {ci}): got {output[ri][ci]}, expected {expected[ri][ci]}")
                        break
                else:
                    continue
                break
    
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
