"""
ARC Puzzle 5b984a1a Solver

Pattern: Draw tiled diagonal lines from corner markers
- Find marker pixels at corners (different from background)
- Fill background with yellow (4)
- Draw diagonals from each marker corner with periodic tiling
- Period = 2 * min(H-1, W-1)
"""

def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])
    
    # Find background color (center pixel)
    bg_color = grid[H // 2][W // 2]
    
    # Find corners with markers
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    marker_corners = []
    marker_color = None
    
    for r, c in corners:
        if grid[r][c] != bg_color:
            marker_corners.append((r, c))
            marker_color = grid[r][c]
    
    # Period for tiling diagonals
    period = 2 * min(H - 1, W - 1)
    
    # Get diagonal values of marker corners mod period
    corner_diffs = set((mr - mc) % period for mr, mc in marker_corners)
    corner_sums = set((mr + mc) % period for mr, mc in marker_corners)
    
    # Create output grid filled with yellow (4)
    output = [[4 for _ in range(W)] for _ in range(H)]
    
    # Draw tiled diagonals
    for r in range(H):
        for c in range(W):
            # Cell is on marker diagonal if its r-c or r+c matches a corner mod period
            on_diag = ((r - c) % period in corner_diffs) or ((r + c) % period in corner_sums)
            if on_diag:
                output[r][c] = marker_color
    
    return output


if __name__ == "__main__":
    import json
    
    # Load task
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    task = data['5b984a1a']
    
    # Test on all training examples
    all_passed = True
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        passed = result == expected
        all_passed = all_passed and passed
        
        print(f"Train {i}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"  Input shape: {len(inp)}x{len(inp[0])}")
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Result shape: {len(result)}x{len(result[0])}")
            # Show first difference
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  First diff at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
                        break
                else:
                    continue
                break
    
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
