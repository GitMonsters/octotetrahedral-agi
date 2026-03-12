import json
import numpy as np

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Transform grid by connecting 2x2 blocks of value 2 to all 2x2 blocks of value 8
    using rectilinear paths filled with value 7.
    
    Algorithm:
    1. Find all 2x2 blocks of value 2 (sources) and value 8 (targets)
    2. Build a minimum spanning tree (MST) connecting all targets to the source
    3. For each MST edge, draw a rectilinear path:
       - Vertical segment from source row to target row at source column
       - Horizontal segment from source column to target column at target row
    
    This creates an optimal network of paths connecting all blocks.
    """
    result = [row[:] for row in grid]
    arr = np.array(result)
    h, w = arr.shape
    
    def find_blocks(value):
        """Find all 2x2 blocks of a given value"""
        blocks = []
        for r in range(h - 1):
            for c in range(w - 1):
                if (arr[r, c] == value and arr[r, c+1] == value and
                    arr[r+1, c] == value and arr[r+1, c+1] == value):
                    blocks.append((r, c))
        return blocks
    
    blocks_8 = sorted(find_blocks(8))
    blocks_2 = find_blocks(2)
    
    if not blocks_2 or not blocks_8:
        return result
    
    src_r, src_c = blocks_2[0]
    
    # Build MST using Prim's algorithm
    in_tree = {(src_r, src_c)}
    all_nodes = {(src_r, src_c)} | set(blocks_8)
    edges = []
    
    while len(in_tree) < len(all_nodes):
        min_dist = float('inf')
        best_edge = None
        
        for from_node in in_tree:
            fr, fc = from_node
            for to_node in all_nodes:
                if to_node not in in_tree:
                    tr, tc = to_node
                    dist = abs(tr - fr) + abs(tc - fc)
                    if dist < min_dist:
                        min_dist = dist
                        best_edge = (from_node, to_node)
        
        if best_edge is None:
            break
        
        (fr, fc), (tr, tc) = best_edge
        edges.append(best_edge)
        in_tree.add((tr, tc))
    
    # Draw MST edges using rectilinear routing
    for (fr, fc), (tr, tc) in edges:
        # Vertical segment: from source row to target row at source column
        min_r = min(fr, tr)
        max_r = max(fr, tr)
        for r in range(min_r + 2, max_r):
            result[r][fc] = 7
            if fc + 1 < w:
                result[r][fc + 1] = 7
        
        # Horizontal segment: from source column to target column at target row
        min_c = min(fc, tc)
        max_c = max(fc, tc)
        for c in range(min_c + 2, max_c):
            result[tr][c] = 7
            if tr + 1 < h:
                result[tr + 1][c] = 7
    
    return result

if __name__ == "__main__":
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/3490cc26.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
