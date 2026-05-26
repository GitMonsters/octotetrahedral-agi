"""Solver for ARC puzzle 4341642d.

Pattern: Each connected component of non-background cells is recolored
based on its number of concave corners:
  1 concave corner  -> color 1
  2 concave corners -> color 7
  3+ concave corners -> color 0
"""

from collections import deque, Counter
from typing import List


def transform(grid: List[List[int]]) -> List[List[int]]:
    H = len(grid)
    W = len(grid[0])

    # Find background color (most frequent)
    freq = Counter()
    for row in grid:
        freq.update(row)
    bg = freq.most_common(1)[0][0]

    # Create output grid initialized to background
    out = [[bg] * W for _ in range(H)]

    # Find connected components via BFS
    visited = [[False] * W for _ in range(H)]
    for sr in range(H):
        for sc in range(W):
            if visited[sr][sc] or grid[sr][sc] == bg:
                continue
            color = grid[sr][sc]
            cells = []
            q = deque([(sr, sc)])
            visited[sr][sc] = True
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                        visited[nr][nc] = True
                        q.append((nr, nc))

            # Count concave corners
            cell_set = set(cells)
            concave = 0
            for r, c in cells:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    adj_r = (r + dr, c) in cell_set
                    adj_c = (r, c + dc) in cell_set
                    diag = (r + dr, c + dc) in cell_set
                    if adj_r and adj_c and not diag:
                        concave += 1

            # Map concave corner count to output color
            if concave == 1:
                new_color = 1
            elif concave == 2:
                new_color = 7
            else:
                new_color = 0

            for r, c in cells:
                out[r][c] = new_color

    return out


def test():
    import json
    import os
    
    # Try multiple possible paths for the data file
    possible_paths = [
        '/tmp/rearc45/4341642d.json',
        '4341642d.json',
        'rearc45/4341642d.json',
        '../rearc45/4341642d.json'
    ]
    
    data_file = None
    for path in possible_paths:
        if os.path.exists(path):
            data_file = path
            break
    
    if data_file is None:
        print(f"Data file not found. Searched paths:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nDemo: Running transform on a simple test case...")
        
        # Create a simple test case
        test_grid = [
            [8, 8, 8, 8, 8],
            [8, 1, 1, 8, 8],
            [8, 1, 1, 8, 8],
            [8, 8, 8, 8, 8]
        ]
        result = transform(test_grid)
        print("Input grid:")
        for row in test_grid:
            print(row)
        print("\nOutput grid:")
        for row in result:
            print(row)
        return
    
    print(f"Loading data from: {data_file}")
    data = json.load(open(data_file))
    for i, pair in enumerate(data['train']):
        result = transform(pair['input'])
        expected = pair['output']
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")


if __name__ == '__main__':
    test()
