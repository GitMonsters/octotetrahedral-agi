"""
HYPOTHESIS FOR ARC TASK 20a5584e:

Looking at the visualizations and patterns, I can see:

1. Each grid has a background color (most frequent)
2. There's an original shape/pattern made of some non-background color
3. There are isolated dots of color 1 (blue) scattered around
4. The transformation replicates the original shape at each isolated 1 position

The key insight: Find isolated 1s and replicate the existing pattern around them.

Let me verify this hypothesis by checking each training pair manually:

PAIR 1:
- Background: 1 (blue) 
- Original shape: 7 (orange) at positions (16,18), (17,17), (18,17), (18,18)
- This forms an L-shape pattern
- Isolated 1s: Since background is 1, we need to find 1s that are NOT part of the background
- Wait... this doesn't make sense. Let me re-examine.

Actually, looking at the visualization again:
- The visualization shows blue background with orange shapes
- But the analysis shows background as 1 (blue)
- This means the "isolated 1s" concept is wrong for this pair

Let me look at this differently...

HYPOTHESIS V2:
Look at the original pattern, then look at where new copies appear in the output.
The transformation places copies of the original pattern at specific locations.

Let me manually check what changes between input and output for each pair.
"""

import json
import numpy as np

def simple_hypothesis_check():
    with open('/Users/evanpieser/apr12_tasks/20a5584e.json', 'r') as f:
        task = json.load(f)
    
    for i, pair in enumerate(task['train']):
        print(f"\n=== PAIR {i+1} ===")
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        # Find what's different
        changes = []
        for r in range(len(input_grid)):
            for c in range(len(input_grid[0])):
                if input_grid[r, c] != output_grid[r, c]:
                    changes.append((r, c, input_grid[r, c], output_grid[r, c]))
        
        print(f"Changes: {len(changes)}")
        if len(changes) < 50:
            for change in changes:
                print(f"  {change}")
        else:
            print(f"  Too many changes, showing first 10:")
            for change in changes[:10]:
                print(f"    {change}")

if __name__ == "__main__":
    simple_hypothesis_check()