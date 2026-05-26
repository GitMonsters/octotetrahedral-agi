import json

# Load task data  
with open('/Users/evanpieser/apr12_tasks/76d965ef.json', 'r') as f:
    data = json.load(f)

def fresh_analysis():
    """Take a completely fresh look at what's happening"""
    
    print("=== FRESH ANALYSIS OF 76d965ef ===")
    
    # Look at Train 0 - the simplest failing case
    example = data['train'][0]
    input_grid = example['input']
    output_grid = example['output']
    
    print("TRAIN 0 - Input 8x8 → Output 16x16")
    
    # Print both grids side by side for comparison
    print("\nInput grid:")
    for r in range(len(input_grid)):
        row = input_grid[r]
        print(f"Row {r:2d}: {row}")
    
    print(f"\nOutput grid:")
    for r in range(len(output_grid)):
        row = output_grid[r]
        print(f"Row {r:2d}: {row}")
    
    # NEW INSIGHT: What if the transformation is simpler?
    # What if it's about taking the pattern and creating a specific tiling?
    
    print("\n=== NEW HYPOTHESIS ===")
    print("What if the rule is: Take the pattern and tile it in specific positions?")
    
    # Extract the 6x6 pattern
    pattern = []
    for r in range(2, 8):  # rows 2-7 
        row = [input_grid[r][c] for c in range(2, 8)]  # cols 2-7
        pattern.append(row)
    
    print(f"\nExtracted 6x6 pattern:")
    for i, row in enumerate(pattern):
        print(f"  {i}: {row}")
    
    # Now let's see if I can find the pattern in specific positions in the output
    print(f"\nLooking for pattern occurrences in output...")
    
    # Try different starting positions
    positions_to_check = [
        (0, 0),   # top-left
        (0, 6),   # top-middle  
        (0, 10),  # top-right (if it fits)
        (6, 0),   # middle-left
        (10, 10), # bottom-right (if it fits)
    ]
    
    for start_r, start_c in positions_to_check:
        print(f"\nChecking position ({start_r}, {start_c}):")
        found_match = True
        for pr in range(6):
            for pc in range(6):
                out_r, out_c = start_r + pr, start_c + pc
                if out_r < len(output_grid) and out_c < len(output_grid[0]):
                    expected = pattern[pr][pc]
                    actual = output_grid[out_r][out_c]
                    if expected != actual:
                        found_match = False
                        break
            if not found_match:
                break
        
        if found_match:
            print(f"  ✓ PATTERN FOUND at ({start_r}, {start_c})")
        else:
            print(f"  ✗ No match")

fresh_analysis()