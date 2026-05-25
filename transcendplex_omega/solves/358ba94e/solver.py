from collections import deque, Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Extract the 5x5 region with a unique zero count.
    The input contains multiple 5x5 rectangular regions of a single non-zero color.
    The output is the region whose count of zero cells appears exactly once across all regions.
    """
    # Find the non-zero color
    colors = set()
    for row in grid:
        colors.update(row)
    colors.discard(0)
    
    if not colors:
        return [[0] * 5 for _ in range(5)]
    
    color = list(colors)[0]
    
    # Find all rectangular regions of this color using BFS
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    rects = []
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == color and not visited[r][c]:
                # BFS to find bounds of connected region
                q = deque([(r, c)])
                visited[r][c] = True
                min_r, max_r = r, r
                min_c, max_c = c, c
                
                while q:
                    cr, cc = q.popleft()
                    min_r, max_r = min(min_r, cr), max(max_r, cr)
                    min_c, max_c = min(min_c, cc), max(max_c, cc)
                    
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                            if grid[nr][nc] == color and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                
                rects.append({'bounds': (min_r, max_r, min_c, max_c)})
    
    # Count zero cells in each region
    zero_counts = []
    for rect in rects:
        r_min, r_max, c_min, c_max = rect['bounds']
        zeros = sum(1 for r in range(r_min, r_max + 1)
                    for c in range(c_min, c_max + 1)
                    if grid[r][c] == 0)
        zero_counts.append(zeros)
    
    # Find the region with a unique zero count
    counter = Counter(zero_counts)
    unique_counts = [c for c, freq in counter.items() if freq == 1]
    
    if unique_counts:
        # Extract the region with unique zero count
        for i, rect in enumerate(rects):
            if zero_counts[i] in unique_counts:
                r_min, r_max, c_min, c_max = rect['bounds']
                return [grid[r][c_min:c_max + 1] for r in range(r_min, r_max + 1)]
    
    # Fallback (should not reach here with valid input)
    return [[0] * 5 for _ in range(5)]


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/358ba94e.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
