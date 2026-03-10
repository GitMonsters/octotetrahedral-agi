def solve(grid):
    """Find the 3x3 block with the most distinct colors."""
    rows = len(grid)
    cols = len(grid[0])

    blocks = []
    visited = [[False] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                stack = [(r, c)]
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    if 0 <= cr < rows and 0 <= cc < cols and not visited[cr][cc] and grid[cr][cc] != 0:
                        visited[cr][cc] = True
                        cells.append((cr, cc))
                        stack.extend([(cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)])
                if cells:
                    min_r = min(x[0] for x in cells)
                    min_c = min(x[1] for x in cells)
                    max_r = max(x[0] for x in cells)
                    max_c = max(x[1] for x in cells)
                    block = []
                    for br in range(min_r, max_r + 1):
                        block.append([grid[br][bc] for bc in range(min_c, max_c + 1)])
                    colors = set()
                    for row in block:
                        for v in row:
                            if v != 0:
                                colors.add(v)
                    blocks.append((len(colors), block))

    blocks.sort(key=lambda x: x[0], reverse=True)
    return blocks[0][1]


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
