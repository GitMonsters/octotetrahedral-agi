import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Find two vertical line pairs and extend diagonal rays from all 4 corners.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Find vertical lines
    segments_by_color = {}
    for col in range(cols):
        row = 0
        while row < rows:
            if grid[row][col] != 0:
                color = grid[row][col]
                r_start = row
                while row < rows and grid[row][col] == color:
                    row += 1
                r_end = row - 1
                if color not in segments_by_color:
                    segments_by_color[color] = []
                segments_by_color[color].append((r_start, r_end, col))
            else:
                row += 1
    
    # Group into pairs
    line_pairs = {}
    for color, segs in segments_by_color.items():
        segs.sort(key=lambda x: (x[0], x[2]))
        used = set()
        for i, (r_start, r_end, col1) in enumerate(segs):
            if i in used:
                continue
            for j, (r_start2, r_end2, col2) in enumerate(segs[i+1:], i+1):
                if j in used or r_start != r_start2 or r_end != r_end2:
                    continue
                if color not in line_pairs:
                    line_pairs[color] = []
                line_pairs[color].append((r_start, r_end, min(col1, col2), max(col1, col2)))
                used.add(i)
                used.add(j)
                break
    
    output = [[0] * cols for _ in range(rows)]
    layer_cells = {}
    
    for color in [3, 8]:
        if color not in line_pairs:
            continue
        
        layer_cells[color] = set()
        
        for r_start, r_end, c_min, c_max in line_pairs[color]:
            # Add the line itself
            for r in range(r_start, r_end + 1):
                for c in [c_min, c_max]:
                    layer_cells[color].add((r, c))
            
            # Extend 45-degree diagonals from all 4 corners
            # Top-left from (r_start, c_min)
            r, c = r_start - 1, c_min - 1
            while r >= 0 and c >= 0:
                layer_cells[color].add((r, c))
                r -= 1
                c -= 1
            
            # Top-right from (r_start, c_max)
            r, c = r_start - 1, c_max + 1
            while r >= 0 and c < cols:
                layer_cells[color].add((r, c))
                r -= 1
                c += 1
            
            # Bottom-left from (r_end, c_min)
            r, c = r_end + 1, c_min - 1
            while r < rows and c >= 0:
                layer_cells[color].add((r, c))
                r += 1
                c -= 1
            
            # Bottom-right from (r_end, c_max)
            r, c = r_end + 1, c_max + 1
            while r < rows and c < cols:
                layer_cells[color].add((r, c))
                r += 1
                c += 1
    
    # Write to output
    for color in [3, 8]:
        if color in layer_cells:
            for r, c in layer_cells[color]:
                output[r][c] = color
    
    # Mark intersections
    if 3 in layer_cells and 8 in layer_cells:
        for r, c in layer_cells[3] & layer_cells[8]:
            output[r][c] = 6
    
    return output


if __name__ == '__main__':
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/58e15b12.json', 'r') as f:
        task = json.load(f)
    
    print("Testing training examples:")
    all_pass = True
    for i, example in enumerate(task['train']):
        result = solve(example['input'])
        expected = example['output']
        
        if result == expected:
            print(f"  Training {i}: PASS")
        else:
            print(f"  Training {i}: FAIL")
            all_pass = False
    
    if all_pass:
        print("All training examples PASS!")
    else:
        print("Some examples FAILED")
