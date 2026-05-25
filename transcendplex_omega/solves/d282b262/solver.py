"""
ARC-AGI Task d282b262 Solver

Pattern: The grid contains rectangular checkerboard blocks. Blocks that share
row ranges are grouped together. Each group is pushed to the right edge of the
grid: the rightmost block in the group aligns to the grid's right edge, and
other blocks in the group are placed immediately to its left at their shared
rows, preserving original row positions.
"""

from collections import defaultdict


def solve(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])

    # Find blocks (connected components of non-zero cells)
    visited = [[False] * W for _ in range(H)]
    blocks = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0 and not visited[r][c]:
                cells = []
                queue = [(r, c)]
                visited[r][c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                min_r = min(r for r, c in cells)
                max_r = max(r for r, c in cells)
                min_c = min(c for r, c in cells)
                max_c = max(c for r, c in cells)

                blocks.append({
                    'row': min_r, 'col': min_c,
                    'height': max_r - min_r + 1,
                    'width': max_c - min_c + 1,
                    'data': [[grid[r][c] for c in range(min_c, max_c + 1)]
                             for r in range(min_r, max_r + 1)]
                })

    # Union-Find to group blocks that overlap in row ranges
    n = len(blocks)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    def shares_rows(a: dict, b: dict) -> bool:
        return (a['row'] <= b['row'] + b['height'] - 1 and
                b['row'] <= a['row'] + a['height'] - 1)

    for i in range(n):
        for j in range(i + 1, n):
            if shares_rows(blocks[i], blocks[j]):
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # Place each group: rightmost block → right edge, others adjacent left
    output = [[0] * W for _ in range(H)]

    for group_indices in groups.values():
        gb = [blocks[i] for i in group_indices]
        gb.sort(key=lambda b: b['col'], reverse=True)

        # Anchor = rightmost block, placed at grid's right edge
        anchor = gb[0]
        anchor['out_col'] = W - anchor['width']

        # BFS from anchor: each connected block is placed to the left
        placed = {id(anchor)}
        q = [anchor]
        while q:
            cur = q.pop(0)
            for blk in gb:
                if id(blk) not in placed and shares_rows(cur, blk):
                    blk['out_col'] = cur['out_col'] - blk['width']
                    placed.add(id(blk))
                    q.append(blk)

        # Write blocks to output at computed positions
        for blk in gb:
            for r in range(blk['height']):
                for c in range(blk['width']):
                    output[blk['row'] + r][blk['out_col'] + c] = blk['data'][r][c]

    return output
