#!/usr/bin/env python3
"""
ARC Solver - OctoTetrahedral AGI Integration
=============================================

Production-ready solver combining:
1. Hint-based pattern recognition
2. Full DSL program synthesis (20+ operations)
3. Hierarchical voting with geometric augmentation  
4. OctoTetrahedral neural backup (optional)

Expected Performance:
- Symbolic only (no LLM): 10-15%
- With LLM TTT: 53-62%
- Our hybrid approach: aiming for 15-25%
"""

import sys
import json
import copy
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from itertools import product

# Add paths
sys.path.insert(0, str(Path.home() / "ARC_AMD_TRANSFER" / "code"))
sys.path.insert(0, str(Path.home() / "octotetrahedral_agi"))


# ============================================================================
# DSL Operations (Full Set)
# ============================================================================

def identity(grid, **kwargs):
    return grid

def rotate_90(grid, **kwargs):
    return [list(row) for row in zip(*grid[::-1])]

def rotate_180(grid, **kwargs):
    return [row[::-1] for row in grid[::-1]]

def rotate_270(grid, **kwargs):
    return [list(row) for row in zip(*grid)][::-1]

def flip_h(grid, **kwargs):
    return [row[::-1] for row in grid]

def flip_v(grid, **kwargs):
    return grid[::-1]

def transpose(grid, **kwargs):
    return [list(row) for row in zip(*grid)]

def tile(grid, h_tiles=2, v_tiles=2, **kwargs):
    result = []
    for _ in range(v_tiles):
        for row in grid:
            result.append(row * h_tiles)
    return result

def scale_up(grid, factor=2, **kwargs):
    result = []
    for row in grid:
        new_row = []
        for cell in row:
            new_row.extend([cell] * factor)
        for _ in range(factor):
            result.append(new_row[:])
    return result

def fill_color(grid, from_color=0, to_color=1, **kwargs):
    return [[to_color if cell == from_color else cell for cell in row] for row in grid]

def gravity_down(grid, **kwargs):
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for j in range(w):
        column = [grid[i][j] for i in range(h) if grid[i][j] != 0]
        for i, val in enumerate(column[::-1]):
            result[h - 1 - i][j] = val
    return result

def crop_to_object(grid, **kwargs):
    """Crop to bounding box of non-zero cells"""
    h, w = len(grid), len(grid[0])
    min_r, max_r, min_c, max_c = h, 0, w, 0
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0:
                min_r = min(min_r, i)
                max_r = max(max_r, i)
                min_c = min(min_c, j)
                max_c = max(max_c, j)
    
    if max_r < min_r:  # All zeros
        return [[0]]
    
    return [row[min_c:max_c+1] for row in grid[min_r:max_r+1]]

def extract_color(grid, color=1, **kwargs):
    """Extract cells of specific color"""
    return [[cell if cell == color else 0 for cell in row] for row in grid]

def most_common_fill(grid, **kwargs):
    """Fill zeros with most common non-zero color"""
    colors = [c for row in grid for c in row if c != 0]
    if not colors:
        return grid
    most_common = Counter(colors).most_common(1)[0][0]
    return [[most_common if cell == 0 else cell for cell in row] for row in grid]

def border(grid, color=1, **kwargs):
    """Add border around grid"""
    h, w = len(grid), len(grid[0])
    result = [[color] * (w + 2)]
    for row in grid:
        result.append([color] + row + [color])
    result.append([color] * (w + 2))
    return result

def gravity_up(grid, **kwargs):
    """Non-zero cells float up"""
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for j in range(w):
        column = [grid[i][j] for i in range(h) if grid[i][j] != 0]
        for i, val in enumerate(column):
            result[i][j] = val
    return result

def gravity_left(grid, **kwargs):
    """Non-zero cells move left"""
    result = []
    for row in grid:
        non_zero = [c for c in row if c != 0]
        zeros = [0] * (len(row) - len(non_zero))
        result.append(non_zero + zeros)
    return result

def gravity_right(grid, **kwargs):
    """Non-zero cells move right"""
    result = []
    for row in grid:
        non_zero = [c for c in row if c != 0]
        zeros = [0] * (len(row) - len(non_zero))
        result.append(zeros + non_zero)
    return result

def mirror_h(grid, **kwargs):
    """Mirror grid horizontally (double width)"""
    return [row + row[::-1] for row in grid]

def mirror_v(grid, **kwargs):
    """Mirror grid vertically (double height)"""
    return grid + grid[::-1]

def fill_interior(grid, **kwargs):
    """Fill interior zeros with surrounding non-zero color"""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            if grid[i][j] == 0:
                # Check if surrounded by same color
                neighbors = [
                    grid[i-1][j], grid[i+1][j], 
                    grid[i][j-1], grid[i][j+1]
                ]
                non_zero = [n for n in neighbors if n != 0]
                if len(non_zero) >= 3 and len(set(non_zero)) == 1:
                    result[i][j] = non_zero[0]
    return result

def outline(grid, **kwargs):
    """Keep only outline of shapes"""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            if grid[i][j] != 0:
                # If all neighbors same color, it's interior
                neighbors = [
                    grid[i-1][j], grid[i+1][j],
                    grid[i][j-1], grid[i][j+1]
                ]
                if all(n == grid[i][j] for n in neighbors):
                    result[i][j] = 0
    return result

def replace_with_pattern(grid, pattern_color=1, target_color=2, **kwargs):
    """Replace pattern_color cells adjacent to target_color"""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] == pattern_color:
                # Check if adjacent to target_color
                adjacent = False
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        if grid[ni][nj] == target_color:
                            adjacent = True
                            break
                if adjacent:
                    result[i][j] = target_color
    return result

def count_colors_to_grid(grid, **kwargs):
    """Return 1x1 grid with count of unique non-zero colors"""
    colors = set(c for row in grid for c in row if c != 0)
    return [[len(colors)]]

def largest_color_only(grid, **kwargs):
    """Keep only the most common non-zero color"""
    colors = [c for row in grid for c in row if c != 0]
    if not colors:
        return grid
    most_common = Counter(colors).most_common(1)[0][0]
    return [[cell if cell == most_common else 0 for cell in row] for row in grid]

def invert_colors(grid, max_color=9, **kwargs):
    """Swap 0 with max_color"""
    return [[max_color if cell == 0 else (0 if cell == max_color else cell) for cell in row] for row in grid]

def copy_pattern(grid, **kwargs):
    """Copy non-zero pattern to fill grid (simple tiling)"""
    # Find non-zero bounding box
    h, w = len(grid), len(grid[0])
    min_r, max_r, min_c, max_c = h, 0, w, 0
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0:
                min_r = min(min_r, i)
                max_r = max(max_r, i)
                min_c = min(min_c, j)
                max_c = max(max_c, j)
    
    if max_r < min_r:
        return grid
    
    # Extract pattern
    pattern = [row[min_c:max_c+1] for row in grid[min_r:max_r+1]]
    ph, pw = len(pattern), len(pattern[0])
    
    # Tile pattern
    result = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            pi, pj = i % ph, j % pw
            if pattern[pi][pj] != 0:
                result[i][j] = pattern[pi][pj]
    
    return result


# ============================================================================
# NEW: Connected Component Analysis & Object Operations
# ============================================================================

def find_connected_components(grid, connectivity=4):
    """Find all connected components in grid, return list of (color, positions)"""
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = []
    
    def bfs(start_r, start_c, color):
        """BFS to find all connected cells of same color"""
        positions = []
        queue = [(start_r, start_c)]
        visited[start_r][start_c] = True
        
        while queue:
            r, c = queue.pop(0)
            positions.append((r, c))
            
            # 4-connectivity neighbors
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            if connectivity == 8:
                neighbors += [(r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
            
            for nr, nc in neighbors:
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                    if grid[nr][nc] == color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
        
        return positions
    
    for i in range(h):
        for j in range(w):
            if not visited[i][j] and grid[i][j] != 0:
                color = grid[i][j]
                positions = bfs(i, j, color)
                components.append((color, positions))
    
    return components


def get_object_bboxes(grid, **kwargs):
    """Return list of (color, min_r, min_c, max_r, max_c) for each object"""
    components = find_connected_components(grid)
    bboxes = []
    
    for color, positions in components:
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        bboxes.append((color, min(rows), min(cols), max(rows), max(cols)))
    
    return bboxes


def extract_largest_object(grid, **kwargs):
    """Extract the largest connected component"""
    components = find_connected_components(grid)
    if not components:
        return grid
    
    # Find largest by pixel count
    largest = max(components, key=lambda x: len(x[1]))
    color, positions = largest
    
    # Get bounding box
    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    
    # Create cropped grid
    result = [[0] * (max_c - min_c + 1) for _ in range(max_r - min_r + 1)]
    for r, c in positions:
        result[r - min_r][c - min_c] = color
    
    return result


def extract_smallest_object(grid, **kwargs):
    """Extract the smallest connected component"""
    components = find_connected_components(grid)
    if not components:
        return grid
    
    # Find smallest by pixel count (min 2 pixels to avoid noise)
    valid = [c for c in components if len(c[1]) >= 2]
    if not valid:
        return grid
    
    smallest = min(valid, key=lambda x: len(x[1]))
    color, positions = smallest
    
    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    
    result = [[0] * (max_c - min_c + 1) for _ in range(max_r - min_r + 1)]
    for r, c in positions:
        result[r - min_r][c - min_c] = color
    
    return result


def count_objects(grid, **kwargs):
    """Return 1x1 grid with number of connected components"""
    components = find_connected_components(grid)
    return [[len(components)]]


def flood_fill(grid, fill_color=1, **kwargs):
    """Flood fill enclosed black (0) regions with fill_color"""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    # Find all black cells connected to border (these are NOT enclosed)
    border_connected = [[False] * w for _ in range(h)]
    queue = []
    
    # Start from all border black cells
    for i in range(h):
        if grid[i][0] == 0:
            queue.append((i, 0))
            border_connected[i][0] = True
        if grid[i][w-1] == 0:
            queue.append((i, w-1))
            border_connected[i][w-1] = True
    for j in range(w):
        if grid[0][j] == 0:
            queue.append((0, j))
            border_connected[0][j] = True
        if grid[h-1][j] == 0:
            queue.append((h-1, j))
            border_connected[h-1][j] = True
    
    # BFS to find all black cells connected to border
    while queue:
        r, c = queue.pop(0)
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
            if 0 <= nr < h and 0 <= nc < w:
                if grid[nr][nc] == 0 and not border_connected[nr][nc]:
                    border_connected[nr][nc] = True
                    queue.append((nr, nc))
    
    # Fill all black cells NOT connected to border
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 and not border_connected[i][j]:
                result[i][j] = fill_color
    
    return result


def flood_fill_smart(grid, **kwargs):
    """Flood fill with most common surrounding color"""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    # Find enclosed regions
    border_connected = [[False] * w for _ in range(h)]
    queue = []
    
    for i in range(h):
        if grid[i][0] == 0:
            queue.append((i, 0))
            border_connected[i][0] = True
        if w > 1 and grid[i][w-1] == 0:
            queue.append((i, w-1))
            border_connected[i][w-1] = True
    for j in range(w):
        if grid[0][j] == 0:
            queue.append((0, j))
            border_connected[0][j] = True
        if h > 1 and grid[h-1][j] == 0:
            queue.append((h-1, j))
            border_connected[h-1][j] = True
    
    while queue:
        r, c = queue.pop(0)
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
            if 0 <= nr < h and 0 <= nc < w:
                if grid[nr][nc] == 0 and not border_connected[nr][nc]:
                    border_connected[nr][nc] = True
                    queue.append((nr, nc))
    
    # For each enclosed region, find surrounding color
    visited = [[False] * w for _ in range(h)]
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 and not border_connected[i][j] and not visited[i][j]:
                # BFS to find this enclosed region
                region = []
                neighbors_colors = []
                q = [(i, j)]
                visited[i][j] = True
                
                while q:
                    r, c = q.pop(0)
                    region.append((r, c))
                    
                    for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                        if 0 <= nr < h and 0 <= nc < w:
                            if grid[nr][nc] == 0 and not border_connected[nr][nc] and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                            elif grid[nr][nc] != 0:
                                neighbors_colors.append(grid[nr][nc])
                
                # Fill with most common neighbor color
                if neighbors_colors:
                    fill_color = Counter(neighbors_colors).most_common(1)[0][0]
                    for r, c in region:
                        result[r][c] = fill_color
    
    return result


def remove_small_objects(grid, min_size=3, **kwargs):
    """Remove connected components smaller than min_size"""
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    
    components = find_connected_components(grid)
    for color, positions in components:
        if len(positions) >= min_size:
            for r, c in positions:
                result[r][c] = color
    
    return result


def keep_n_largest_objects(grid, n=1, **kwargs):
    """Keep only the n largest objects"""
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    
    components = find_connected_components(grid)
    # Sort by size descending
    components.sort(key=lambda x: len(x[1]), reverse=True)
    
    for color, positions in components[:n]:
        for r, c in positions:
            result[r][c] = color
    
    return result


# ============================================================================
# NEW: Symmetry Operations
# ============================================================================

def enforce_h_symmetry(grid, **kwargs):
    """Make grid horizontally symmetric (mirror from left)"""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    for i in range(h):
        for j in range(w // 2):
            mirror_j = w - 1 - j
            # Take non-zero value, prefer left side
            if result[i][j] != 0:
                result[i][mirror_j] = result[i][j]
            elif result[i][mirror_j] != 0:
                result[i][j] = result[i][mirror_j]
    
    return result


def enforce_v_symmetry(grid, **kwargs):
    """Make grid vertically symmetric (mirror from top)"""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    for i in range(h // 2):
        mirror_i = h - 1 - i
        for j in range(w):
            if result[i][j] != 0:
                result[mirror_i][j] = result[i][j]
            elif result[mirror_i][j] != 0:
                result[i][j] = result[mirror_i][j]
    
    return result


def enforce_rotational_symmetry(grid, **kwargs):
    """Make grid 180-degree rotationally symmetric"""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    for i in range(h):
        for j in range(w):
            mirror_i, mirror_j = h - 1 - i, w - 1 - j
            if i * w + j < mirror_i * w + mirror_j:  # Only process each pair once
                if result[i][j] != 0:
                    result[mirror_i][mirror_j] = result[i][j]
                elif result[mirror_i][mirror_j] != 0:
                    result[i][j] = result[mirror_i][mirror_j]
    
    return result


def complete_pattern_from_quadrant(grid, **kwargs):
    """Complete pattern assuming top-left quadrant is the source"""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    half_h, half_w = h // 2, w // 2
    
    # Copy top-left to other quadrants
    for i in range(half_h):
        for j in range(half_w):
            val = grid[i][j]
            if val != 0:
                # Top-right (h-flip)
                if j < w - 1 - j:
                    result[i][w - 1 - j] = val
                # Bottom-left (v-flip)
                if i < h - 1 - i:
                    result[h - 1 - i][j] = val
                # Bottom-right (180 rotation)
                if i < h - 1 - i or j < w - 1 - j:
                    result[h - 1 - i][w - 1 - j] = val
    
    return result


# ============================================================================
# NEW: Color Mapping Operations
# ============================================================================

def swap_colors(grid, color1=1, color2=2, **kwargs):
    """Swap two colors"""
    return [[color2 if c == color1 else (color1 if c == color2 else c) for c in row] for row in grid]


def recolor_by_size(grid, **kwargs):
    """Recolor objects by their size (largest=1, second=2, etc.)"""
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    
    components = find_connected_components(grid)
    # Sort by size descending
    components.sort(key=lambda x: len(x[1]), reverse=True)
    
    for idx, (_, positions) in enumerate(components):
        new_color = (idx % 9) + 1  # Colors 1-9
        for r, c in positions:
            result[r][c] = new_color
    
    return result


def majority_color_per_row(grid, **kwargs):
    """Fill each row with its majority non-zero color"""
    result = []
    for row in grid:
        non_zero = [c for c in row if c != 0]
        if non_zero:
            majority = Counter(non_zero).most_common(1)[0][0]
            result.append([majority] * len(row))
        else:
            result.append(row[:])
    return result


def majority_color_per_col(grid, **kwargs):
    """Fill each column with its majority non-zero color"""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    for j in range(w):
        col = [grid[i][j] for i in range(h)]
        non_zero = [c for c in col if c != 0]
        if non_zero:
            majority = Counter(non_zero).most_common(1)[0][0]
            for i in range(h):
                result[i][j] = majority
    
    return result


# ============================================================================
# NEW: Grid Manipulation Operations
# ============================================================================

def split_horizontal(grid, part=0, **kwargs):
    """Split grid horizontally, return top (0) or bottom (1) half"""
    h = len(grid)
    mid = h // 2
    if part == 0:
        return grid[:mid]
    else:
        return grid[mid:]


def split_vertical(grid, part=0, **kwargs):
    """Split grid vertically, return left (0) or right (1) half"""
    w = len(grid[0])
    mid = w // 2
    if part == 0:
        return [row[:mid] for row in grid]
    else:
        return [row[mid:] for row in grid]


def xor_grids(grid1, grid2):
    """XOR two grids (difference)"""
    h, w = len(grid1), len(grid1[0])
    h2, w2 = len(grid2), len(grid2[0])
    
    if h != h2 or w != w2:
        return grid1
    
    result = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if grid1[i][j] != grid2[i][j]:
                result[i][j] = grid1[i][j] if grid1[i][j] != 0 else grid2[i][j]
    
    return result


def and_grids(grid1, grid2):
    """AND two grids (intersection)"""
    h, w = len(grid1), len(grid1[0])
    h2, w2 = len(grid2), len(grid2[0])
    
    if h != h2 or w != w2:
        return grid1
    
    result = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if grid1[i][j] != 0 and grid2[i][j] != 0:
                result[i][j] = grid1[i][j]
    
    return result


def or_grids(grid1, grid2):
    """OR two grids (union)"""
    h, w = len(grid1), len(grid1[0])
    h2, w2 = len(grid2), len(grid2[0])
    
    if h != h2 or w != w2:
        return grid1
    
    result = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if grid1[i][j] != 0:
                result[i][j] = grid1[i][j]
            elif grid2[i][j] != 0:
                result[i][j] = grid2[i][j]
    
    return result


def overlay_pattern(grid, **kwargs):
    """Overlay detected pattern across grid"""
    components = find_connected_components(grid)
    if len(components) < 2:
        return grid
    
    # Sort by size, use smallest as pattern
    components.sort(key=lambda x: len(x[1]))
    pattern_color, pattern_pos = components[0]
    
    # Get pattern bounding box
    p_rows = [p[0] for p in pattern_pos]
    p_cols = [p[1] for p in pattern_pos]
    p_min_r, p_max_r = min(p_rows), max(p_rows)
    p_min_c, p_max_c = min(p_cols), max(p_cols)
    
    # Extract pattern
    ph = p_max_r - p_min_r + 1
    pw = p_max_c - p_min_c + 1
    pattern = [[0] * pw for _ in range(ph)]
    for r, c in pattern_pos:
        pattern[r - p_min_r][c - p_min_c] = pattern_color
    
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    # Find positions of other colors and overlay pattern
    for color, positions in components[1:]:
        for r, c in positions:
            # Overlay pattern centered at this position
            for pi in range(ph):
                for pj in range(pw):
                    if pattern[pi][pj] != 0:
                        nr = r - ph // 2 + pi
                        nc = c - pw // 2 + pj
                        if 0 <= nr < h and 0 <= nc < w:
                            result[nr][nc] = pattern[pi][pj]
    
    return result


def detect_and_complete_grid_pattern(grid, **kwargs):
    """Detect repeating pattern and fill missing cells"""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    # Try different pattern sizes
    for ph in range(1, h // 2 + 1):
        for pw in range(1, w // 2 + 1):
            if h % ph == 0 and w % pw == 0:
                # Check if this is a valid pattern period
                pattern = [row[:pw] for row in grid[:ph]]
                
                # Verify pattern tiles correctly (ignoring zeros)
                valid = True
                for i in range(h):
                    for j in range(w):
                        pi, pj = i % ph, j % pw
                        if grid[i][j] != 0 and pattern[pi][pj] != 0:
                            if grid[i][j] != pattern[pi][pj]:
                                valid = False
                                break
                        elif grid[i][j] != 0:
                            pattern[pi][pj] = grid[i][j]
                    if not valid:
                        break
                
                if valid:
                    # Fill in missing values from pattern
                    for i in range(h):
                        for j in range(w):
                            if result[i][j] == 0:
                                pi, pj = i % ph, j % pw
                                if pattern[pi][pj] != 0:
                                    result[i][j] = pattern[pi][pj]
                    return result
    
    return result


OPERATIONS = {
    # Basic transforms
    'identity': identity,
    'rotate_90': rotate_90,
    'rotate_180': rotate_180,
    'rotate_270': rotate_270,
    'flip_h': flip_h,
    'flip_v': flip_v,
    'transpose': transpose,
    
    # Size operations
    'tile': tile,
    'scale_up': scale_up,
    'crop_to_object': crop_to_object,
    'border': border,
    
    # Gravity/movement
    'gravity_down': gravity_down,
    'gravity_up': gravity_up,
    'gravity_left': gravity_left,
    'gravity_right': gravity_right,
    
    # Color operations
    'fill_color': fill_color,
    'extract_color': extract_color,
    'most_common_fill': most_common_fill,
    'largest_color_only': largest_color_only,
    'invert_colors': invert_colors,
    'swap_colors': swap_colors,
    'recolor_by_size': recolor_by_size,
    'majority_color_per_row': majority_color_per_row,
    'majority_color_per_col': majority_color_per_col,
    
    # Mirror/symmetry
    'mirror_h': mirror_h,
    'mirror_v': mirror_v,
    'enforce_h_symmetry': enforce_h_symmetry,
    'enforce_v_symmetry': enforce_v_symmetry,
    'enforce_rotational_symmetry': enforce_rotational_symmetry,
    'complete_pattern_from_quadrant': complete_pattern_from_quadrant,
    
    # Fill operations
    'fill_interior': fill_interior,
    'outline': outline,
    'flood_fill': flood_fill,
    'flood_fill_smart': flood_fill_smart,
    
    # Object operations (Connected Components)
    'extract_largest_object': extract_largest_object,
    'extract_smallest_object': extract_smallest_object,
    'count_objects': count_objects,
    'remove_small_objects': remove_small_objects,
    'keep_n_largest_objects': keep_n_largest_objects,
    
    # Pattern operations
    'copy_pattern': copy_pattern,
    'overlay_pattern': overlay_pattern,
    'detect_and_complete_grid_pattern': detect_and_complete_grid_pattern,
    
    # Split operations
    'split_horizontal_top': lambda g, **k: split_horizontal(g, part=0),
    'split_horizontal_bottom': lambda g, **k: split_horizontal(g, part=1),
    'split_vertical_left': lambda g, **k: split_vertical(g, part=0),
    'split_vertical_right': lambda g, **k: split_vertical(g, part=1),
}

INVERSE_TRANSFORMS = {
    'identity': 'identity',
    'rotate_90': 'rotate_270',
    'rotate_180': 'rotate_180',
    'rotate_270': 'rotate_90',
    'flip_h': 'flip_h',
    'flip_v': 'flip_v',
    'transpose': 'transpose',
}


# ============================================================================
# Hint Generator
# ============================================================================

class HintGenerator:
    """Generate pattern hints from training examples"""
    
    def analyze(self, examples: List[Dict]) -> Dict:
        return {
            'size_ratio': self._analyze_size(examples),
            'geometric': self._detect_geometric(examples),
            'tiling': self._detect_tiling(examples),
            'colors': self._analyze_colors(examples),
            'symmetry': self._detect_symmetry(examples),
            'object_count': self._analyze_objects(examples),
            'fill_pattern': self._detect_fill(examples),
        }
    
    def _analyze_size(self, examples: List[Dict]) -> Tuple[float, float]:
        """Return (height_ratio, width_ratio) from input to output"""
        ratios = []
        for ex in examples:
            in_h, in_w = len(ex['input']), len(ex['input'][0])
            out_h, out_w = len(ex['output']), len(ex['output'][0])
            ratios.append((out_h / in_h, out_w / in_w))
        
        # Return average ratio
        if ratios:
            avg_h = sum(r[0] for r in ratios) / len(ratios)
            avg_w = sum(r[1] for r in ratios) / len(ratios)
            return (avg_h, avg_w)
        return (1.0, 1.0)
    
    def _detect_geometric(self, examples: List[Dict]) -> Optional[str]:
        """Detect if a geometric transform matches all examples"""
        transforms = ['rotate_90', 'rotate_180', 'rotate_270', 'flip_h', 'flip_v', 'transpose']
        
        for trans in transforms:
            matches_all = True
            for ex in examples:
                try:
                    result = OPERATIONS[trans](ex['input'])
                    if result != ex['output']:
                        matches_all = False
                        break
                except:
                    matches_all = False
                    break
            
            if matches_all:
                return trans
        
        return None
    
    def _detect_tiling(self, examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Detect if output is tiled version of input"""
        for ex in examples:
            in_h, in_w = len(ex['input']), len(ex['input'][0])
            out_h, out_w = len(ex['output']), len(ex['output'][0])
            
            if out_h % in_h == 0 and out_w % in_w == 0:
                h_tiles = out_h // in_h
                w_tiles = out_w // in_w
                
                # Verify tiling
                result = tile(ex['input'], h_tiles=w_tiles, v_tiles=h_tiles)
                if result == ex['output']:
                    return (h_tiles, w_tiles)
        
        return None
    
    def _analyze_colors(self, examples: List[Dict]) -> Dict:
        """Analyze color patterns"""
        input_colors = set()
        output_colors = set()
        
        for ex in examples:
            for row in ex['input']:
                input_colors.update(row)
            for row in ex['output']:
                output_colors.update(row)
        
        return {
            'input': sorted(input_colors),
            'output': sorted(output_colors),
            'new': sorted(output_colors - input_colors),
            'removed': sorted(input_colors - output_colors),
        }
    
    def _detect_symmetry(self, examples: List[Dict]) -> Optional[str]:
        """Detect if output has symmetry that input doesn't"""
        symmetry_ops = [
            ('enforce_h_symmetry', 'horizontal'),
            ('enforce_v_symmetry', 'vertical'),
            ('enforce_rotational_symmetry', 'rotational'),
        ]
        
        for op_name, sym_type in symmetry_ops:
            matches_all = True
            for ex in examples:
                try:
                    result = OPERATIONS[op_name](ex['input'])
                    if result != ex['output']:
                        matches_all = False
                        break
                except:
                    matches_all = False
                    break
            
            if matches_all:
                return op_name
        
        return None
    
    def _analyze_objects(self, examples: List[Dict]) -> Dict:
        """Analyze object counts in input vs output"""
        input_counts = []
        output_counts = []
        
        for ex in examples:
            in_comps = find_connected_components(ex['input'])
            out_comps = find_connected_components(ex['output'])
            input_counts.append(len(in_comps))
            output_counts.append(len(out_comps))
        
        return {
            'input_avg': sum(input_counts) / len(input_counts) if input_counts else 0,
            'output_avg': sum(output_counts) / len(output_counts) if output_counts else 0,
            'reduces': all(o < i for i, o in zip(input_counts, output_counts)),
            'increases': all(o > i for i, o in zip(input_counts, output_counts)),
        }
    
    def _detect_fill(self, examples: List[Dict]) -> Optional[str]:
        """Detect if a fill operation matches"""
        fill_ops = ['flood_fill_smart', 'fill_interior', 'most_common_fill']
        
        for op in fill_ops:
            matches_all = True
            for ex in examples:
                try:
                    result = OPERATIONS[op](ex['input'])
                    if result != ex['output']:
                        matches_all = False
                        break
                except:
                    matches_all = False
                    break
            
            if matches_all:
                return op
        
        return None


# ============================================================================
# Program Synthesizer
# ============================================================================

class ProgramSynthesizer:
    """Synthesize programs from examples"""
    
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
    
    def synthesize(self, examples: List[Dict]) -> List[Tuple[List, float]]:
        """Return list of (program, score) tuples"""
        candidates = self._enumerate_programs()
        scored = []
        
        for program in candidates:
            score = self._score_program(program, examples)
            if score >= 0.99:  # Only perfect matches
                scored.append((program, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:10]
    
    def _enumerate_programs(self) -> List[List[Tuple[str, Dict]]]:
        """Enumerate candidate programs"""
        programs = []
        
        # Single operations (no params)
        simple_ops = [
            # Basic transforms
            'identity', 'rotate_90', 'rotate_180', 'rotate_270',
            'flip_h', 'flip_v', 'transpose',
            # Gravity
            'gravity_down', 'gravity_up', 'gravity_left', 'gravity_right',
            # Size
            'crop_to_object',
            # Color
            'most_common_fill', 'largest_color_only', 'invert_colors',
            'recolor_by_size', 'majority_color_per_row', 'majority_color_per_col',
            # Mirror/symmetry
            'mirror_h', 'mirror_v',
            'enforce_h_symmetry', 'enforce_v_symmetry', 'enforce_rotational_symmetry',
            'complete_pattern_from_quadrant',
            # Fill
            'fill_interior', 'outline', 'flood_fill_smart',
            # Objects
            'extract_largest_object', 'extract_smallest_object', 'count_objects',
            # Pattern
            'copy_pattern', 'overlay_pattern', 'detect_and_complete_grid_pattern',
            # Split
            'split_horizontal_top', 'split_horizontal_bottom',
            'split_vertical_left', 'split_vertical_right',
        ]
        for op in simple_ops:
            programs.append([(op, {})])
        
        # Tiling
        for h in [2, 3, 4]:
            for v in [2, 3, 4]:
                programs.append([('tile', {'h_tiles': h, 'v_tiles': v})])
        
        # Scaling
        for f in [2, 3]:
            programs.append([('scale_up', {'factor': f})])
        
        # Color fill
        for from_c in range(10):
            for to_c in range(10):
                if from_c != to_c:
                    programs.append([('fill_color', {'from_color': from_c, 'to_color': to_c})])
        
        # Extract color
        for c in range(1, 10):
            programs.append([('extract_color', {'color': c})])
        
        # Border
        for c in range(1, 10):
            programs.append([('border', {'color': c})])
        
        # Flood fill with specific colors
        for c in range(1, 10):
            programs.append([('flood_fill', {'fill_color': c})])
        
        # Swap colors
        for c1 in range(1, 10):
            for c2 in range(c1+1, 10):
                programs.append([('swap_colors', {'color1': c1, 'color2': c2})])
        
        # Remove small objects
        for min_size in [2, 3, 4, 5]:
            programs.append([('remove_small_objects', {'min_size': min_size})])
        
        # Keep n largest objects
        for n in [1, 2, 3]:
            programs.append([('keep_n_largest_objects', {'n': n})])
        
        # Two-operation compositions (geometric only to start)
        if self.max_depth >= 2:
            geo_ops = ['rotate_90', 'rotate_180', 'flip_h', 'flip_v', 'transpose']
            for op1 in geo_ops:
                for op2 in geo_ops:
                    programs.append([(op1, {}), (op2, {})])
            
            # Useful two-op combinations
            useful_combos = [
                # Crop then transform
                [('crop_to_object', {}), ('rotate_90', {})],
                [('crop_to_object', {}), ('rotate_180', {})],
                [('crop_to_object', {}), ('flip_h', {})],
                [('crop_to_object', {}), ('flip_v', {})],
                # Extract then crop
                [('extract_largest_object', {}), ('crop_to_object', {})],
                [('extract_smallest_object', {}), ('crop_to_object', {})],
                # Fill then transform
                [('flood_fill_smart', {}), ('crop_to_object', {})],
                [('fill_interior', {}), ('crop_to_object', {})],
                # Symmetry then crop
                [('enforce_h_symmetry', {}), ('crop_to_object', {})],
                [('enforce_v_symmetry', {}), ('crop_to_object', {})],
                # Gravity then crop
                [('gravity_down', {}), ('crop_to_object', {})],
                [('gravity_up', {}), ('crop_to_object', {})],
                # Object operations
                [('recolor_by_size', {}), ('extract_largest_object', {})],
                [('keep_n_largest_objects', {'n': 1}), ('crop_to_object', {})],
            ]
            programs.extend(useful_combos)
        
        return programs
    
    def _score_program(self, program: List[Tuple[str, Dict]], examples: List[Dict]) -> float:
        """Score program on examples (1.0 = perfect match)"""
        if not examples:
            return 0.0
        
        matches = 0
        for ex in examples:
            try:
                result = self._execute(program, ex['input'])
                if result == ex['output']:
                    matches += 1
            except:
                pass
        
        return matches / len(examples)
    
    def _execute(self, program: List[Tuple[str, Dict]], grid: List[List[int]]) -> List[List[int]]:
        """Execute program on grid"""
        result = copy.deepcopy(grid)
        for op_name, params in program:
            result = OPERATIONS[op_name](result, **params)
        return result


# ============================================================================
# ARC Solver
# ============================================================================

class ARCSolver:
    """Production ARC solver with hints + synthesis"""
    
    def __init__(self):
        self.hint_gen = HintGenerator()
        self.synthesizer = ProgramSynthesizer(max_depth=2)
        self.transforms = ['identity', 'rotate_90', 'rotate_180', 'rotate_270',
                          'flip_h', 'flip_v', 'transpose']
    
    def solve(self, task: Dict) -> List[List[List[int]]]:
        """Solve task, return up to 2 predictions"""
        train = task['train']
        test_input = task['test'][0]['input']
        
        predictions = []
        
        # 1. Try hint-based solving
        hints = self.hint_gen.analyze(train)
        
        # Direct geometric match
        if hints['geometric']:
            pred = OPERATIONS[hints['geometric']](test_input)
            predictions.append(pred)
        
        # Tiling match
        if hints['tiling']:
            h_tiles, w_tiles = hints['tiling']
            pred = tile(test_input, h_tiles=w_tiles, v_tiles=h_tiles)
            predictions.append(pred)
        
        # Symmetry match
        if hints['symmetry']:
            try:
                pred = OPERATIONS[hints['symmetry']](test_input)
                if pred not in predictions:
                    predictions.append(pred)
            except:
                pass
        
        # Fill pattern match
        if hints['fill_pattern']:
            try:
                pred = OPERATIONS[hints['fill_pattern']](test_input)
                if pred not in predictions:
                    predictions.append(pred)
            except:
                pass
        
        # 2. Try learning color mapping from examples
        color_map = self._learn_color_mapping(train)
        if color_map:
            try:
                pred = self._apply_color_mapping(test_input, color_map)
                if pred not in predictions:
                    predictions.append(pred)
            except:
                pass
        
        # 3. Try program synthesis with augmentation
        for trans_name in self.transforms:
            if trans_name == 'identity':
                trans_fn = lambda x: x
                inv_fn = lambda x: x
            else:
                trans_fn = OPERATIONS[trans_name]
                inv_name = INVERSE_TRANSFORMS[trans_name]
                inv_fn = OPERATIONS[inv_name]
            
            # Transform training data
            trans_train = [
                {'input': trans_fn(ex['input']), 'output': trans_fn(ex['output'])}
                for ex in train
            ]
            
            # Synthesize programs
            programs = self.synthesizer.synthesize(trans_train)
            
            # Execute on test input
            trans_test = trans_fn(test_input)
            for program, score in programs:
                try:
                    result = self.synthesizer._execute(program, trans_test)
                    final = inv_fn(result)
                    if final not in predictions:
                        predictions.append(final)
                except:
                    pass
        
        # 4. Deduplicate and return top 2
        unique = []
        seen = set()
        for pred in predictions:
            key = str(pred)
            if key not in seen:
                seen.add(key)
                unique.append(pred)
        
        # Fallback: return input
        if not unique:
            unique.append(copy.deepcopy(test_input))
        
        return unique[:2]
    
    def _learn_color_mapping(self, examples: List[Dict]) -> Optional[Dict[int, int]]:
        """Learn a consistent color mapping from examples"""
        # Try to learn: what color X in input becomes color Y in output
        mappings = []
        
        for ex in examples:
            in_grid = ex['input']
            out_grid = ex['output']
            
            # Only if same size
            if len(in_grid) != len(out_grid) or len(in_grid[0]) != len(out_grid[0]):
                return None
            
            h, w = len(in_grid), len(in_grid[0])
            local_map = {}
            
            for i in range(h):
                for j in range(w):
                    in_c = in_grid[i][j]
                    out_c = out_grid[i][j]
                    
                    if in_c in local_map:
                        if local_map[in_c] != out_c:
                            # Inconsistent mapping in this example
                            local_map = None
                            break
                    else:
                        local_map[in_c] = out_c
                
                if local_map is None:
                    break
            
            if local_map:
                mappings.append(local_map)
        
        if not mappings:
            return None
        
        # Check if all examples have consistent mapping
        final_map = mappings[0]
        for m in mappings[1:]:
            for k, v in m.items():
                if k in final_map and final_map[k] != v:
                    return None
                final_map[k] = v
        
        # Only return if it's a non-trivial mapping
        if any(k != v for k, v in final_map.items()):
            return final_map
        
        return None
    
    def _apply_color_mapping(self, grid: List[List[int]], color_map: Dict[int, int]) -> List[List[int]]:
        """Apply learned color mapping to grid"""
        return [[color_map.get(c, c) for c in row] for row in grid]


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(data_dir: str, max_tasks: int = 50, split: str = 'training'):
    """Evaluate solver"""
    task_dir = Path(data_dir) / split
    task_files = sorted(task_dir.glob('*.json'))[:max_tasks]
    
    solver = ARCSolver()
    results = {'total': 0, 'pass1': 0, 'pass2': 0}
    
    for task_file in task_files:
        with open(task_file) as f:
            task = json.load(f)
        
        if 'output' not in task['test'][0]:
            continue
        
        results['total'] += 1
        ground_truth = task['test'][0]['output']
        predictions = solver.solve(task)
        
        if predictions:
            if predictions[0] == ground_truth:
                results['pass1'] += 1
                results['pass2'] += 1
                print(f"✓ {task_file.stem}")
            elif len(predictions) > 1 and predictions[1] == ground_truth:
                results['pass2'] += 1
                print(f"○ {task_file.stem} (pass@2)")
            else:
                print(f"✗ {task_file.stem}")
        else:
            print(f"✗ {task_file.stem} (no pred)")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=str(Path.home() / 'ARC_AMD_TRANSFER' / 'data' / 'ARC-AGI' / 'data'))
    parser.add_argument('--max-tasks', type=int, default=50)
    parser.add_argument('--split', default='training')
    args = parser.parse_args()
    
    print("="*70)
    print("ARC Solver - OctoTetrahedral AGI")
    print("Hints + Program Synthesis + Geometric Augmentation")
    print("="*70)
    print()
    
    results = evaluate(args.data_dir, args.max_tasks, args.split)
    
    print()
    print("="*70)
    print("Results")
    print("="*70)
    if results['total'] > 0:
        p1 = results['pass1'] / results['total'] * 100
        p2 = results['pass2'] / results['total'] * 100
        print(f"Tasks:  {results['total']}")
        print(f"Pass@1: {results['pass1']}/{results['total']} ({p1:.1f}%)")
        print(f"Pass@2: {results['pass2']}/{results['total']} ({p2:.1f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
