import json
from collections import deque


def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    twos = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                twos.append((r, c))

    def bfs(sr, sc):
        dist = [[-1] * cols for _ in range(rows)]
        dist[sr][sc] = 0
        q = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and dist[nr][nc] == -1 and grid[nr][nc] != 1:
                    dist[nr][nc] = dist[r][c] + 1
                    q.append((nr, nc))
        return dist

    dist_a = bfs(*twos[0])
    dist_b = bfs(*twos[1])
    shortest = dist_a[twos[1][0]][twos[1][1]]

    out = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and dist_a[r][c] >= 0 and dist_b[r][c] >= 0:
                if dist_a[r][c] + dist_b[r][c] == shortest:
                    out[r][c] = 4

    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/b15fca0b.json") as f:
        task = json.load(f)

    all_pass = True
    for i, example in enumerate(task["train"]):
        result = solve(example["input"])
        if result == example["output"]:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            for r, (got, exp) in enumerate(zip(result, example["output"])):
                if got != exp:
                    print(f"  Row {r}: got {got}")
                    print(f"       exp {exp}")
            all_pass = False

    for i, example in enumerate(task["test"]):
        result = solve(example["input"])
        if result == example["output"]:
            print(f"Test  {i}: PASS")
        else:
            print(f"Test  {i}: FAIL")
            for r, (got, exp) in enumerate(zip(result, example["output"])):
                if got != exp:
                    print(f"  Row {r}: got {got}")
                    print(f"       exp {exp}")
            all_pass = False

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
