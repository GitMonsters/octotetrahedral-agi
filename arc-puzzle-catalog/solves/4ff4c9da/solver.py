import sys
import json
from typing import List, Set, Tuple


def find_clusters(positions):
    """Find clusters of consecutive positions (allowing gaps up to 2)."""
    if not positions:
        return []
    
    sorted_pos = sorted(positions)
    clusters = []
    current_cluster = [sorted_pos[0]]
    
    for i in range(1, len(sorted_pos)):
        if sorted_pos[i] - sorted_pos[i-1] <= 2:
            current_cluster.append(sorted_pos[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_pos[i]]
    
    clusters.append(current_cluster)
    return clusters


def find_spacing(clusters):
    """Find the spacing between clusters."""
    if len(clusters) < 2:
        return None
    
    # Calculate center of each cluster
    centers = []
    for cluster in clusters:
        center = (cluster[0] + cluster[-1]) // 2
        centers.append(center)
    
    # Find spacing (distance between consecutive centers)
    if len(centers) >= 2:
        return centers[1] - centers[0]
    
    return None


def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    Find pattern of 8s marked in input and replicate across grid using detected spacing.
    
    The input contains 8s that mark certain positions. The output should have
    these 8s replicated at regular spacing intervals detected from the clusters.
    """
    if not grid:
        return grid
    
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if height == 0 or width == 0:
        return grid
    
    # Find all positions with 8 in input
    input_positions = set()
    for i in range(height):
        for j in range(width):
            if grid[i][j] == 8:
                input_positions.add((i, j))
    
    if not input_positions:
        return [row[:] for row in grid]
    
    # Extract rows and columns with 8s
    input_rows = sorted(set(r for r, c in input_positions))
    input_cols = sorted(set(c for r, c in input_positions))
    
    # Find clusters and spacing
    row_clusters = find_clusters(input_rows)
    col_clusters = find_clusters(input_cols)
    
    row_spacing = find_spacing(row_clusters)
    col_spacing = find_spacing(col_clusters)
    
    # If we can't determine spacing, use a default (try different values)
    if not row_spacing:
        row_spacing = height // 2
    if not col_spacing:
        col_spacing = width // 2
    
    # Determine the "base" positions by normalizing input positions to first cluster
    # Map all input 8 positions to their base tile coordinates
    marked_base_positions = set()
    
    if row_clusters and col_clusters:
        # Get offsets within the first cluster
        first_row_cluster_start = row_clusters[0][0]
        first_col_cluster_start = col_clusters[0][0]
        
        for r, c in input_positions:
            # Normalize to position within the repeating unit
            base_r = (r - first_row_cluster_start) % row_spacing if row_spacing else r % row_spacing
            base_c = (c - first_col_cluster_start) % col_spacing if col_spacing else c % col_spacing
            marked_base_positions.add((base_r, base_c, first_row_cluster_start % row_spacing, first_col_cluster_start % col_spacing))
    
    # Create output
    output = [row[:] for row in grid]
    
    # Apply marks to all corresponding positions
    if marked_base_positions:
        for r, c, offset_r, offset_c in marked_base_positions:
            for i in range(height):
                for j in range(width):
                    if (i - offset_r) % row_spacing == r and (j - offset_c) % col_spacing == c:
                        output[i][j] = 8
    
    return output


if __name__ == "__main__":
    task_path = sys.argv[1] if len(sys.argv) > 1 else "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/4ff4c9da.json"
    task_path = task_path.replace("~", "/Users/evanpieser")
    
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    all_pass = True
    for idx, example in enumerate(task['train']):
        result = solve(example['input'])
        expected = example['output']
        
        if result == expected:
            print(f"Training Example {idx + 1}: PASS")
        else:
            print(f"Training Example {idx + 1}: FAIL")
            all_pass = False
            
            # Show first difference
            for i in range(len(result)):
                for j in range(len(result[i])):
                    if result[i][j] != expected[i][j]:
                        print(f"  First diff at ({i}, {j}): got {result[i][j]}, expected {expected[i][j]}")
                        break
                if any(result[i][j] != expected[i][j] for j in range(len(result[i]))):
                    break
    
    if all_pass:
        print("\nAll training examples PASS!")
    else:
        print("\nSome examples FAILED")
