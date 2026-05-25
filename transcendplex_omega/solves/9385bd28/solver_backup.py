#!/usr/bin/env python3

import json
import sys
import copy

def solve(grid):
    """
    Pattern analysis from examples:
    1. Find disconnected clusters of same colored non-background objects
    2. Use color mapping hints to determine fill color
    3. Create bounding rectangle around SIGNIFICANT clusters only
    4. Fill the BORDER/FRAME of rectangle with mapping color, preserve interior
    """
    
    result = copy.deepcopy(grid)
    height = len(grid)
    width = len(grid[0])
    background = get_background_color(grid)
    
    # Get color mappings from the grid (usually at bottom)
    color_mappings = get_color_mappings(grid, background)
    
    # Find all non-background positions grouped by color
    color_positions = {}
    for r in range(height):
        for c in range(width):
            if grid[r][c] != background:
                color = grid[r][c]
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append((r, c))
    
    # For each color that has a mapping, check if it needs rectangle connection
    for color, mapping_color in color_mappings.items():
        if color in color_positions and len(color_positions[color]) > 1:
            positions = color_positions[color]
            clusters = find_clusters(positions)
            
            # Filter to only significant clusters (size > 1)
            significant_clusters = [c for c in clusters if len(c) > 1]
            
            # If multiple significant disconnected clusters exist, connect them
            if len(significant_clusters) > 1:
                # Get bounding box for significant clusters only
                significant_positions = []
                for cluster in significant_clusters:
                    significant_positions.extend(cluster)
                
                min_r = min(pos[0] for pos in significant_positions)
                max_r = max(pos[0] for pos in significant_positions)
                min_c = min(pos[1] for pos in significant_positions)
                max_c = max(pos[1] for pos in significant_positions)
                
                # Fill rectangle with mapping color, but be more selective
                for r in range(min_r, max_r + 1):
                    for c in range(min_c, max_c + 1):
                        if result[r][c] == background:
                            result[r][c] = mapping_color
    
    return result

def get_background_color(grid):
    """Find the most common color, which is typically the background"""
    color_counts = {}
    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1
    return max(color_counts.items(), key=lambda x: x[1])[0]

def get_color_mappings(grid, background):
    """Find color mappings from bottom rows, left column, or other indicators"""
    height = len(grid)
    width = len(grid[0])
    mappings = {}
    
    # Check bottom rows for horizontal pairs (common pattern)
    for r in range(max(0, height - 3), height):
        for c in range(width - 1):
            left = grid[r][c]
            right = grid[r][c + 1]
            if (left != background and right != background and 
                left != right):
                mappings[left] = right
    
    # Check left columns for vertical pairs
    for r in range(height - 1):
        for c in range(min(3, width)):
            top = grid[r][c] 
            bottom = grid[r + 1][c]
            if (top != background and bottom != background and
                top != bottom):
                mappings[top] = bottom
    
    return mappings

def find_clusters(positions):
    """Find disconnected clusters of positions using flood fill"""
    if not positions:
        return []
    
    positions_set = set(positions)
    clusters = []
    visited = set()
    
    for pos in positions:
        if pos in visited:
            continue
        
        # Start new cluster
        cluster = []
        stack = [pos]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
                
            visited.add(current)
            cluster.append(current)
            
            # Check 4-connected neighbors
            r, c = current
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = (r + dr, c + dc)
                if neighbor in positions_set and neighbor not in visited:
                    stack.append(neighbor)
        
        clusters.append(cluster)
    
    return clusters


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python solver.py <input.json>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    
    # Test on training examples
    all_passed = True
    for i, example in enumerate(data['train']):
        result = solve(example['input'])
        expected = example['output']
        
        if result == expected:
            print(f"Training example {i+1}: PASS")
        else:
            print(f"Training example {i+1}: FAIL")
            all_passed = False
    
    # Test on test examples if available
    if 'test' in data:
        for i, example in enumerate(data['test']):
            result = solve(example['input'])
            if 'output' in example:
                expected = example['output']
                if result == expected:
                    print(f"Test example {i+1}: PASS")
                else:
                    print(f"Test example {i+1}: FAIL")
                    all_passed = False
            else:
                print(f"Test example {i+1}: Generated output")
    
    if all_passed:
        print("All examples passed!")