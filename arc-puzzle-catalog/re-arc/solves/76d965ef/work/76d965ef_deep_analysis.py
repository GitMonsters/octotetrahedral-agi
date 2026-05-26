import json

# Load task data
with open('/Users/evanpieser/apr12_tasks/76d965ef.json', 'r') as f:
    data = json.load(f)

def deep_pattern_analysis():
    """Deep analysis of the pattern transformation"""
    
    print("=== DEEP PATTERN ANALYSIS ===")
    
    # Focus on Train 1 - it shows clearest pattern
    example = data['train'][1]
    input_grid = example['input']
    output_grid = example['output']
    
    # Extract pattern (6x6)
    pattern = [
        [7, 8, 8, 2, 1, 7],
        [7, 8, 8, 2, 1, 1],
        [7, 8, 8, 2, 2, 2],
        [7, 8, 8, 8, 8, 8],
        [7, 8, 8, 8, 8, 8],
        [7, 7, 7, 7, 7, 7]
    ]
    
    print("Pattern:")
    for i, row in enumerate(pattern):
        print(f"  {i}: {row}")
    
    print("\nOutput analysis - looking at 3 horizontal segments:")
    
    for row_idx in range(6):
        output_row = output_grid[row_idx]
        
        seg1 = output_row[0:6]
        seg2 = output_row[6:12] 
        seg3 = output_row[12:18]
        
        print(f"\nRow {row_idx}:")
        print(f"  Segment 1 (0-5):   {seg1}")
        print(f"  Segment 2 (6-11):  {seg2}")
        print(f"  Segment 3 (12-17): {seg3}")
        print(f"  Pattern row {row_idx}:      {pattern[row_idx]}")
        
        # Check matches
        match1 = seg1 == pattern[0]  # Always first pattern row?
        match2 = seg2 == pattern[0]  # Always first pattern row?
        match3 = seg3 == pattern[row_idx]  # Actual pattern row?
        
        print(f"  Seg1==Pat[0]: {match1}, Seg2==Pat[0]: {match2}, Seg3==Pat[{row_idx}]: {match3}")

deep_pattern_analysis()