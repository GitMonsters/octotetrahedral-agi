import json
from collections import deque

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Transform puzzle: Each 7-region marks a 1-region cluster.
    Clusters are matched to 7-regions using greedy matching.
    Colors are assigned using hash of 7-region position.
    """
    grid = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    
    def bfs_cells(r, c, color, visited):
        """Get all cells in a connected region of a specific color"""
        cells = []
        q = deque([(r, c)])
        visited[r][c] = True
        
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                    visited[nr][nc] = True
                    q.append((nr, nc))
        
        return cells
    
    # Find all 7-regions and 1-regions
    visited_7 = [[False] * w for _ in range(h)]
    visited_1 = [[False] * w for _ in range(h)]
    
    seven_regions = []
    one_regions = []
    
    # Extract 7-regions
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 7 and not visited_7[r][c]:
                cells = bfs_cells(r, c, 7, visited_7)
                center_r = sum(rr for rr, cc in cells) / len(cells)
                center_c = sum(cc for rr, cc in cells) / len(cells)
                seven_regions.append((center_r, center_c))
    
    # Extract 1-regions
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 1 and not visited_1[r][c]:
                cells = bfs_cells(r, c, 1, visited_1)
                center_r = sum(rr for rr, cc in cells) / len(cells)
                center_c = sum(cc for rr, cc in cells) / len(cells)
                one_regions.append((center_r, center_c, cells))
    
    # Cluster 1-regions based on spatial proximity
    threshold = 5.5
    clusters = []
    used = set()
    
    for i, (r1, c1, _) in enumerate(one_regions):
        if i in used:
            continue
        cluster_indices = [i]
        used.add(i)
        
        for j, (r2, c2, _) in enumerate(one_regions):
            if j not in used:
                dist = ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5
                if dist < threshold:
                    cluster_indices.append(j)
                    used.add(j)
        
        clusters.append(cluster_indices)
    
    # Build distance matrix for greedy matching
    dist_matrix = []
    cluster_centers = []
    
    for cluster_indices in clusters:
        total_r = sum(one_regions[i][0] for i in cluster_indices)
        total_c = sum(one_regions[i][1] for i in cluster_indices)
        center_r = total_r / len(cluster_indices)
        center_c = total_c / len(cluster_indices)
        cluster_centers.append((center_r, center_c))
        
        row = []
        for seven_r, seven_c in seven_regions:
            dist = (center_r - seven_r) ** 2 + (center_c - seven_c) ** 2
            row.append(dist)
        dist_matrix.append(row)
    
    # Greedy matching: match clusters to 7-regions by distance
    matched_clusters = set()
    matched_sevens = set()
    cluster_to_seven = {}
    
    pairs = []
    for cluster_idx in range(len(clusters)):
        for seven_idx in range(len(seven_regions)):
            pairs.append((dist_matrix[cluster_idx][seven_idx], cluster_idx, seven_idx))
    
    pairs.sort()
    
    for dist, cluster_idx, seven_idx in pairs:
        if cluster_idx not in matched_clusters and seven_idx not in matched_sevens:
            matched_clusters.add(cluster_idx)
            matched_sevens.add(seven_idx)
            cluster_to_seven[cluster_idx] = seven_idx
    
    # Assign colors using hash of 7-region position
    color_palette = [3, 4, 6, 8]
    cluster_colors = {}
    
    for cluster_idx, seven_idx in cluster_to_seven.items():
        seven_r, seven_c = seven_regions[seven_idx]
        # Use hash of 7-region position to select color
        hash_val = (int(seven_r) * 13 + int(seven_c) * 7) % len(color_palette)
        cluster_colors[cluster_idx] = color_palette[hash_val]
    
    # Apply colors to 1-regions based on their cluster
    for cluster_idx, cluster_indices in enumerate(clusters):
        if cluster_idx in cluster_colors:
            color = cluster_colors[cluster_idx]
        else:
            color = 2
        
        for one_idx in cluster_indices:
            _, _, one_cells = one_regions[one_idx]
            for r, c in one_cells:
                grid[r][c] = color
    
    # Remove 7s
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 7:
                grid[r][c] = 0
    
    return grid


if __name__ == "__main__":
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/604001fa.json', 'r') as f:
        task = json.load(f)
    
    print("Testing training examples:")
    for i, example in enumerate(task['train']):
        result = solve(example['input'])
        expected = example['output']
        
        if result == expected:
            print(f"  Example {i+1}: PASS")
        else:
            print(f"  Example {i+1}: FAIL")
