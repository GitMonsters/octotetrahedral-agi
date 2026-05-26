import json
import numpy as np

def verify_hypothesis():
    with open('/Users/evanpieser/apr12_tasks/12191b16.json', 'r') as f:
        task = json.load(f)
    
    print("=== HYPOTHESIS VERIFICATION ===")
    
    for pair_idx, pair in enumerate(task['train']):
        print(f"\n--- Pair {pair_idx + 1} ---")
        
        input_grid = np.array(pair['input'])
        expected_output = np.array(pair['output'])
        
        # Find background color
        unique, counts = np.unique(input_grid, return_counts=True)
        bg_color = unique[np.argmax(counts)]
        
        print(f"Background: {bg_color}")
        
        # Key insight: Let's look at the pattern more carefully
        # It seems like the transformation creates a symmetric frame/border pattern
        # based on the bounding box of the dots
        
        # Find dots and bounding box
        dots = []
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                if input_grid[i, j] != bg_color:
                    dots.append((i, j, input_grid[i, j]))
        
        if not dots:
            continue
            
        rows = [d[0] for d in dots]
        cols = [d[1] for d in dots]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        
        print(f"Dot bounding box: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
        
        # Check the pattern structure
        height, width = expected_output.shape
        center_row, center_col = height // 2, width // 2
        
        print(f"Grid center: ({center_row}, {center_col})")
        
        # Look at the symmetric properties
        # Check if rows with dots create patterns that extend symmetrically
        dot_rows = sorted(set(rows))
        dot_cols = sorted(set(cols))
        
        print(f"Rows with dots: {dot_rows}")
        print(f"Cols with dots: {dot_cols}")
        
        # For each dot row, check the full pattern in output
        for dot_row in dot_rows:
            output_row = expected_output[dot_row]
            print(f"Row {dot_row} pattern: {list(output_row[:20])}...")
            
            # Check symmetry from center
            mirror_row = height - 1 - dot_row
            if 0 <= mirror_row < height:
                mirror_output_row = expected_output[mirror_row]
                print(f"Mirror row {mirror_row} pattern: {list(mirror_output_row[:20])}...")
        
        print(f"Verification: Dots preserved? ", end="")
        dots_preserved = True
        for r, c, color in dots:
            if expected_output[r, c] != color:
                dots_preserved = False
                break
        print("YES" if dots_preserved else "NO")

verify_hypothesis()