#!/usr/bin/env python3
"""
ARC-AGI Task 50f325b5 Solver

Pattern: Find the shape formed by 8s in the input. Wherever 3s form the exact
same shape, replace those 3s with 8s.
"""

import json
import sys
from copy import deepcopy


def get_shape(positions):
    """Get normalized shape relative to top-left corner."""
    if not positions:
        return frozenset()
    min_r = min(r for r, c in positions)
    min_c = min(c for r, c in positions)
    return frozenset((r - min_r, c - min_c) for r, c in positions)


def get_all_rotations_and_reflections(shape):
    """Get all 8 possible orientations of a shape."""
    if not shape:
        return [frozenset()]
    
    coords = list(shape)
    max_r = max(r for r, c in coords)
    max_c = max(c for r, c in coords)
    
    orientations = set()
    
    # Original
    orientations.add(frozenset(coords))
    
    # Rotations
    rot_90 = frozenset((c, max_r - r) for r, c in coords)
    orientations.add(rot_90)
    
    rot_180 = frozenset((max_r - r, max_c - c) for r, c in coords)
    orientations.add(rot_180)
    
    rot_270 = frozenset((max_c - c, r) for r, c in coords)
    orientations.add(rot_270)
    
    # Reflections
    flip_h = frozenset((r, max_c - c) for r, c in coords)
    orientations.add(flip_h)
    
    flip_v = frozenset((max_r - r, c) for r, c in coords)
    orientations.add(flip_v)
    
    diag_1 = frozenset((c, r) for r, c in coords)
    orientations.add(diag_1)
    
    diag_2 = frozenset((max_c - c, max_r - r) for r, c in coords)
    orientations.add(diag_2)
    
    return list(orientations)


def try_match_shape_at(grid, target_shape, value, base_r, base_c):
    """Check if target_shape can be matched at position (base_r, base_c) with given value."""
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    for dr, dc in target_shape:
        r, c = base_r + dr, base_c + dc
        if r < 0 or r >= height or c < 0 or c >= width:
            return False
        if grid[r][c] != value:
            return False
    return True


def find_all_shape_matches(grid, target_shape, value):
    """Find all positions where target_shape matches the given value."""
    if not target_shape:
        return []
    
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    max_dr = max(dr for dr, dc in target_shape)
    max_dc = max(dc for dr, dc in target_shape)
    
    matches = []
    for base_r in range(height - max_dr):
        for base_c in range(width - max_dc):
            if try_match_shape_at(grid, target_shape, value, base_r, base_c):
                matches.append((base_r, base_c))
    
    return matches


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solve the ARC puzzle.
    
    1. Find all 8s in the input
    2. Extract the shape formed by the 8s
    3. Get all rotations and reflections of that shape
    4. Find all positions where 3s form any of these shapes
    5. Replace those 3s with 8s
    """
    result = deepcopy(grid)
    height = len(grid)
    if height == 0:
        return result
    width = len(grid[0])
    
    # Find all 8s
    eights = [(r, c) for r in range(height) for c in range(width) if grid[r][c] == 8]
    
    if not eights:
        return result
    
    # Get the shape defined by 8s
    eight_shape = get_shape(eights)
    
    # Get all possible orientations of the 8-shape
    orientations = get_all_rotations_and_reflections(eight_shape)
    
    # For each orientation, find all positions where 3s form that shape
    for orientation in orientations:
        matches = find_all_shape_matches(grid, orientation, 3)
        
        # Replace matched 3s with 8s
        for base_r, base_c in matches:
            for dr, dc in orientation:
                r, c = base_r + dr, base_c + dc
                if 0 <= r < height and 0 <= c < width:
                    result[r][c] = 8
    
    return result


if __name__ == "__main__":
    # Load task JSON
    task_path = sys.argv[1] if len(sys.argv) > 1 else "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/50f325b5.json"
    task_path = task_path.replace("~", "/Users/evanpieser")
    
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    # Test on training examples
    all_pass = True
    for i, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        
        result = solve(input_grid)
        
        # Check if result matches expected
        if result == expected_output:
            print(f"Training example {i + 1}: PASS")
        else:
            print(f"Training example {i + 1}: FAIL")
            all_pass = False
            # Print differences for debugging
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != expected_output[r][c]:
                        print(f"  Mismatch at ({r}, {c}): got {result[r][c]}, expected {expected_output[r][c]}")
    
    if all_pass:
        print("\nAll training examples PASSED!")
    else:
        print("\nSome training examples FAILED!")
        sys.exit(1)
