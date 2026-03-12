import json
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    Color swap puzzle: Mark certain 0-cells with color 3 based on pattern.
    The rule: Find all connected regions of 0s. In each region, identify
    cross/diamond patterns and mark them with color 3.
    """
    output = [row[:] for row in grid]
    
    def get_zero_regions():
        """Find all connected regions of 0s"""
        visited = set()
        regions = []
        
        def dfs(r, c, region):
            if (r, c) in visited or r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
                return
            if grid[r][c] != 0:
                return
            visited.add((r, c))
            region.append((r, c))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                dfs(r + dr, c + dc, region)
        
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if (r, c) not in visited and grid[r][c] == 0:
                    region = []
                    dfs(r, c, region)
                    regions.append(region)
        
        return regions
    
    def find_diamonds_in_region(region):
        """
        Find + cross patterns in a 0-region.
        Cross pattern: center at (r, c) with cells up, down, left, right all being 0.
        """
        region_set = set(region)
        crosses = []
        
        # For each cell in the region, check if it's a + cross center
        for r, c in region:
            # Check for + cross pattern (center + 4 cardinal neighbors)
            plus_cells = [(r, c), (r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            if all(cell in region_set for cell in plus_cells):
                crosses.append(plus_cells)
        
        return crosses
    
    # Find all 0-regions
    regions = get_zero_regions()
    
    # For each region, find diamonds and mark them
    for region in regions:
        diamonds = find_diamonds_in_region(region)
        for diamond in diamonds:
            for r, c in diamond:
                output[r][c] = 3
    
    return output


if __name__ == "__main__":
    import sys
    
    # Load the puzzle
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/7e02026e.json") as f:
        data = json.load(f)
    
    # Test on all training examples
    all_pass = True
    for i, example in enumerate(data['train']):
        result = solve(example['input'])
        expected = example['output']
        
        if result == expected:
            print(f"✓ Training example {i+1} PASSED")
        else:
            print(f"✗ Training example {i+1} FAILED")
            all_pass = False
            
            # Show differences
            diffs = []
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != expected[r][c]:
                        diffs.append((r, c, result[r][c], expected[r][c]))
            
            print(f"  Differences ({len(diffs)} cells):")
            for r, c, got, exp in diffs[:5]:
                print(f"    ({r},{c}): got {got}, expected {exp}")
    
    if all_pass:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        sys.exit(1)
