def solve(grid: list[list[int]]) -> list[list[int]]:
    """Find solid-color rectangles in a noisy grid and return their color layout."""
    rows, cols = len(grid), len(grid[0])
    
    rectangles = []  # (color, top, left, bottom, right)
    
    for color in range(10):
        positions = set()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    positions.add((r, c))
        if not positions:
            continue
        
        remaining = set(positions)
        while remaining:
            start = next(iter(remaining))
            component = set()
            stack = [start]
            while stack:
                pos = stack.pop()
                if pos in component or pos not in remaining:
                    continue
                component.add(pos)
                r, c = pos
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if (nr, nc) in remaining and (nr, nc) not in component:
                        stack.append((nr, nc))
            remaining -= component
            
            min_r = min(r for r, c in component)
            max_r = max(r for r, c in component)
            min_c = min(c for r, c in component)
            max_c = max(c for r, c in component)
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            
            if h >= 3 and w >= 3 and len(component) == h * w:
                rectangles.append((color, min_r, min_c, max_r, max_c))
    
    # Cluster centers to determine grid layout
    centers = [(color, (t+b)/2, (l+r)/2) for color, t, l, b, r in rectangles]
    
    def cluster_by_gaps(values):
        """Split sorted values at the largest natural gaps."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n <= 1:
            return [sum(sorted_vals) / max(n, 1)]
        gaps = [(sorted_vals[i+1] - sorted_vals[i], i) for i in range(n - 1)]
        # Find significant gaps: sort by size, keep those > 2x the minimum non-zero gap
        gap_sizes = sorted(set(g for g, _ in gaps))
        # Use the largest jump in gap sizes to separate "within-cluster" from "between-cluster"
        split_indices = set()
        if gap_sizes:
            sorted_gaps = sorted(gaps, key=lambda x: -x[0])
            # A gap is significant if it's at least 2x the smallest gap or at least 2.5
            min_gap = min(g for g, _ in gaps)
            threshold = max(min_gap * 3, 2.5)
            for g, idx in gaps:
                if g >= threshold:
                    split_indices.add(idx)
        splits = sorted(split_indices)
        result = []
        start = 0
        for s in splits:
            chunk = sorted_vals[start:s+1]
            result.append(sum(chunk) / len(chunk))
            start = s + 1
        chunk = sorted_vals[start:]
        result.append(sum(chunk) / len(chunk))
        return result

    row_clusters = cluster_by_gaps([cy for _, cy, _ in centers])
    col_clusters = cluster_by_gaps([cx for _, _, cx in centers])
    
    output = [[0] * len(col_clusters) for _ in range(len(row_clusters))]
    for color, cy, cx in centers:
        ri = min(range(len(row_clusters)), key=lambda i: abs(row_clusters[i] - cy))
        ci = min(range(len(col_clusters)), key=lambda i: abs(col_clusters[i] - cx))
        output[ri][ci] = color
    
    return output

if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        expected = ex.get('output')
        if expected:
            status = "PASS" if result == expected else "FAIL"
            print(f"Example {i}: {status}")
            if status == "FAIL":
                print(f"  Expected: {expected}")
                print(f"  Got:      {result}")
        else:
            print(f"Example {i}: no expected output → {result}")
