#!/usr/bin/env python3
"""
ARC-AGI puzzle e6de6e8f — Attempt 2

STEP 1-2: Analyze the examples.

Input: 2×12 grid. Two rows of values (0 or 2).
Output: 8×7 grid.

Let me parse the input as segments. The input has 12 columns.
Looking at columns in pairs: the 2 rows form column patterns.

Let me look at the input columns individually:
Example 0 input:
  Row0: [2, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2]
  Row1: [2, 2, 0, 2, 2, 0, 2, 0, 2, 2, 0, 2]

Column patterns (top,bottom):
  c0: (2,2)  c1: (0,2)  c2: (0,0)  c3: (0,2)  c4: (2,2)  c5: (0,0)
  c6: (2,2)  c7: (0,0)  c8: (2,2)  c9: (0,2) c10: (0,0) c11: (2,2)

Separating by (0,0) columns: segments are separated by (0,0).
Segment boundaries at c2, c5, c7, c10.

Actually let me look at it differently. Let me group by the 0,0 separator:
Cols 0-1: (2,2),(0,2) → segment shape: top=[2,0], bot=[2,2]
Col 2: (0,0) separator
Cols 3-4: (0,2),(2,2) → top=[0,2], bot=[2,2]
Col 5: (0,0) separator
Col 6: (2,2) → top=[2], bot=[2]
Col 7: (0,0) separator
Cols 8-9: (2,2),(0,2) → top=[2,0], bot=[2,2]
Col 10: (0,0) separator
Col 11: (2,2) → top=[2], bot=[2]

Hmm, let me try a different grouping. Look at segments separated by column where both are 0.

Actually, let me think about this more carefully by examining the output.

Output 0 (8×7):
Row0: [0, 0, 0, 3, 0, 0, 0]  — green dot at col 3
Row1: [0, 0, 0, 2, 2, 0, 0]
Row2: [0, 0, 0, 2, 2, 0, 0]
Row3: [0, 0, 0, 2, 0, 0, 0]
Row4: [0, 0, 0, 2, 0, 0, 0]
Row5: [0, 0, 0, 2, 2, 0, 0]
Row6: [0, 0, 0, 0, 2, 0, 0]
Row7: [0, 0, 0, 0, 2, 0, 0]

The green dot (3) is always at (0,3). Then below it, pairs of rows show 2s.
Row 1-2: 2s at cols 3,4 → width 2, starting at col 3
Row 3-4: 2s at col 3 → width 1, starting at col 3
Row 5-6: 2s at cols 3,4 → width 2, starting at col 3 (wait, row 6 is col 4 only)

Actually rows are paired:
Rows 1-2: cols 3-4
Rows 3-4: col 3
Rows 5-6: col 3-4 (row5) then col 4 (row6) -- not paired exactly

Let me re-examine. The output is 8 rows. Row 0 has the green marker.
Below that are 7 rows. But we have segments from the input...

Let me look at the segments differently. Each input column (pair of values top,bot):
  If top=2,bot=2: both filled → width=2
  If top=0,bot=2: only bottom → width=1
  If top=2,bot=0: only top → width=1  
  If top=0,bot=0: separator (width=0)

For Example 0: 
  c0:(2,2)→2  c1:(0,2)→1  |  c3:(0,2)→1  c4:(2,2)→2  |  c6:(2,2)→2  |  c8:(2,2)→2  c9:(0,2)→1  |  c11:(2,2)→2

Segments (between 0,0 separators): [2,1], [1,2], [2], [2,1], [2]
Sum of widths: 2+1=3, 1+2=3, 2, 2+1=3, 2 ... hmm

Wait — the output seems to show a path/snake going down. Let me look at the 2-positions:
Row1: cols 3,4 → leftmost=3
Row2: cols 3,4 → leftmost=3
Row3: col 3 → leftmost=3
Row4: col 3 → leftmost=3
Row5: cols 3,4 → leftmost=3
Row6: col 4 → leftmost=4
Row7: col 4 → leftmost=4

It looks like a vertical "snake" that shifts right as it goes down.

Let me reconsider. Each non-separator column in the input encodes one row of the output 
(below the header). The column's pattern (top,bottom) determines width:
  (2,2) → 2 cells wide
  (0,2) or (2,0) → 1 cell wide

And the snake starts at col 3 of the output and moves... 

For example 0, non-separator columns in order: c0,c1,c3,c4,c6,c8,c9,c11
That's 8 columns → but output has 7 data rows (rows 1-7). Hmm, 8 non-sep cols but 7 rows.

Wait, output rows 1-7 = 7 rows. Let me recount non-separator columns for example 0.
(0,0) at c2,c5,c7,c10 → 4 separators → 12-4=8 non-separator columns.

But we only have 7 output rows... 

ALTERNATIVE APPROACH: Maybe the input should be read as segments between (0,0) separators,
and each segment maps to 2 output rows.

Example 0 segments: [c0,c1], [c3,c4], [c6], [c8,c9], [c11]
That's 5 segments. 5 segments × ? ≠ 7 rows.

Let me try yet another approach. Maybe the width of each segment determines something else,
and we have exactly 4 segments (since the last column pair always ends with the grid).

Actually let me re-examine. Maybe segments of the input (separated by 0-columns) become 
vertical sections of the output. Each section is 2 rows high.

5 segments → but only 4 "body" sections (8 rows: 1 header + 7 data OR 4×2=8 rows total minus header).

Hmm. Let me look at this very differently using a snake/path approach.

The output shows a vertical line descending from the green dot. Looking at rightmost 2 
in each row:
Row1: rightmost 2 at col 4 → offset +1 from center(3)
Row2: rightmost at col 4 → +1
Row3: rightmost at col 3 → 0
Row4: rightmost at col 3 → 0
Row5: rightmost at col 4 → +1
Row6: rightmost at col 4 → +1
Row7: rightmost at col 4 → +1

And leftmost 2:
Row1: col 3 → 0
Row2: col 3 → 0
Row3: col 3 → 0
Row4: col 3 → 0
Row5: col 3 → 0
Row6: col 4 → +1
Row7: col 4 → +1

OK let me just write code to carefully analyze all examples and derive the rule.
"""

import json

# Examples
examples = [
    {
        "input": [
            [2, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2],
            [2, 2, 0, 2, 2, 0, 2, 0, 2, 2, 0, 2]
        ],
        "output": [
            [0, 0, 0, 3, 0, 0, 0],
            [0, 0, 0, 2, 2, 0, 0],
            [0, 0, 0, 2, 2, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 2, 0, 0]
        ]
    },
    {
        "input": [
            [0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2],
            [2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 0, 2]
        ],
        "output": [
            [0, 0, 0, 3, 0, 0, 0],
            [0, 0, 2, 2, 0, 0, 0],
            [0, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 2, 0, 0]
        ]
    },
    {
        "input": [
            [2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2],
            [2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 0, 2]
        ],
        "output": [
            [0, 0, 0, 3, 0, 0, 0],
            [0, 0, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0, 2, 2],
            [0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 2]
        ]
    }
]

test_input = [
    [2, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 2],
    [2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2]
]

expected_output = [
    [0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 0],
    [0, 0, 0, 0, 2, 2, 0],
    [0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 2, 0, 0]
]

# STEP 2: Detailed analysis
print("=== STEP 2: Detailed Analysis ===")
for ei, ex in enumerate(examples):
    inp = ex["input"]
    out = ex["output"]
    print(f"\nExample {ei}:")
    
    # Parse input columns
    cols = len(inp[0])
    col_types = []
    for c in range(cols):
        t, b = inp[0][c], inp[1][c]
        col_types.append((t, b))
        print(f"  Col {c}: ({t},{b})", end="")
    print()
    
    # Find segments (separated by (0,0))
    segments = []
    current = []
    for c in range(cols):
        if col_types[c] == (0, 0):
            if current:
                segments.append(current)
                current = []
        else:
            current.append(col_types[c])
    if current:
        segments.append(current)
    
    print(f"  Segments: {segments}")
    print(f"  Num segments: {len(segments)}")
    
    # For each segment, compute: how many 2s in top row, how many in bottom
    for si, seg in enumerate(segments):
        top_count = sum(1 for t, b in seg if t == 2)
        bot_count = sum(1 for t, b in seg if b == 2)
        width = len(seg)
        print(f"  Seg {si}: width={width}, top_2s={top_count}, bot_2s={bot_count}, patterns={seg}")
    
    # Output analysis
    print(f"  Output size: {len(out)}×{len(out[0])}")
    for r in range(len(out)):
        twos = [c for c in range(len(out[r])) if out[r][c] == 2]
        threes = [c for c in range(len(out[r])) if out[r][c] == 3]
        print(f"  Row {r}: 2s at {twos}, 3s at {threes}")

print("\n=== STEP 3-4: Finding the rule ===")

# Let me look at the output more carefully as a "falling" pattern.
# The green dot (3) is at (0, 3) always — center column of 7-wide grid.
# Below it, there's a stem of 2s that descends and potentially shifts.
# 
# Each segment from the input seems to produce 2 rows of output.
# 5 segments * 2 = 10 but we only have 7 data rows... 
# Wait: 4 segments between separators, but segments include start and end.
#
# Let me re-examine: 
# Example 0: 5 segments, output has rows 1-7 = 7 rows
# Example 1: 5 segments... 
# Hmm let me look at segment lengths: each segment has 1 or 2 columns.
# Total non-separator columns across all segments = 7 in each case? Let me check.

for ei, ex in enumerate(examples):
    inp = ex["input"]
    cols = len(inp[0])
    non_sep = sum(1 for c in range(cols) if not (inp[0][c] == 0 and inp[1][c] == 0))
    print(f"Example {ei}: non-separator columns = {non_sep}")

# Let me check: each non-separator column maps to one output row?
# Example 0: non-sep columns (in order): c0,c1,c3,c4,c6,c8,c9,c11 = 8 cols
# But output rows 1-7 = 7 rows. So 8 ≠ 7.

# Hmm wait, let me recount. Maybe I miscounted separators.
# Example 0 row0: [2, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2]
# Example 0 row1: [2, 2, 0, 2, 2, 0, 2, 0, 2, 2, 0, 2]
# Col 0: (2,2) non-sep
# Col 1: (0,2) non-sep
# Col 2: (0,0) sep
# Col 3: (0,2) non-sep
# Col 4: (2,2) non-sep
# Col 5: (0,0) sep
# Col 6: (2,2) non-sep
# Col 7: (0,0) sep
# Col 8: (2,2) non-sep
# Col 9: (0,2) non-sep
# Col 10: (0,0) sep
# Col 11: (2,2) non-sep
# Non-sep: 0,1,3,4,6,8,9,11 = 8 columns.
# But only 7 output rows (rows 1-7).

# Wait! The output has 8 rows total including row 0 (the green dot row).
# Maybe each non-separator column maps to one output row, and the first 
# non-sep column creates the green dot row?

# Let me test: 8 non-sep cols → 8 output rows. Yes!

# Now the question is: what does each non-sep column encode?
# 
# Let me see if the column pattern determines the horizontal extent:
# (2,2) → 2 cells wide (left and right of current position)
# (0,2) → 1 cell wide (right of current position? or just 1 cell?)
# (2,0) → 1 cell wide (left of current position?)
#
# And perhaps the pattern determines direction of shift.

# Let me track the "position" of the 2s in each output row.
# I'll measure the leftmost position of non-zero cells relative to center (col 3).

print("\n=== Position tracking ===")
for ei, ex in enumerate(examples):
    inp = ex["input"]
    out = ex["output"]
    print(f"\nExample {ei}:")
    
    # Get non-separator columns in order
    cols_data = []
    for c in range(len(inp[0])):
        if not (inp[0][c] == 0 and inp[1][c] == 0):
            cols_data.append((inp[0][c], inp[1][c]))
    
    for r in range(len(out)):
        nonzero = [c for c in range(len(out[r])) if out[r][c] != 0]
        if nonzero:
            left = min(nonzero)
            right = max(nonzero)
            width = right - left + 1
        else:
            left = right = width = 0
        
        if r < len(cols_data):
            ct = cols_data[r]
            ct_width = (1 if ct[0] == 2 else 0) + (1 if ct[1] == 2 else 0)
        else:
            ct = None
            ct_width = 0
        
        print(f"  Row {r}: output cols {nonzero}, width={width}, left={left}, input_col=({ct}), input_width={ct_width}")

