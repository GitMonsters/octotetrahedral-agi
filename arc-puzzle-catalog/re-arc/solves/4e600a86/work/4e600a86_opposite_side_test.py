#!/usr/bin/env python3
import json

# Load task data
with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
    task = json.load(f)

def test_opposite_side_hypothesis():
    """Test hypothesis: add cells on opposite side to create horizontal symmetry"""
    train_pairs = task['train']
    
    for i, pair in enumerate(train_pairs):
        print(f"\n{'='*60}")
        print(f"OPPOSITE SIDE HYPOTHESIS - TRAIN PAIR {i+1}")
        print(f"{'='*60}")
        
        input_grid = pair['input']
        output_grid = pair['output']
        h, w = len(input_grid), len(input_grid[0])
        
        # Find background color
        colors = {}
        for row in input_grid:
            for cell in row:
                colors[cell] = colors.get(cell, 0) + 1
        bg_color = max(colors, key=colors.get)
        
        if bg_color == 3:
            print("Background is already 3")
            continue
        
        # Find pattern cells
        pattern_cells = set()
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != bg_color:
                    pattern_cells.add((r, c))
        
        # Find changed cells
        changed_cells = set()
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != output_grid[r][c]:
                    changed_cells.add((r, c))
        
        if not pattern_cells:
            continue
            
        min_r = min(r for r, c in pattern_cells)
        max_r = max(r for r, c in pattern_cells)
        min_c = min(c for r, c in pattern_cells)
        max_c = max(c for r, c in pattern_cells)
        
        # Get pattern spans by row
        pattern_by_row = {}
        for r, c in pattern_cells:
            if r not in pattern_by_row:
                pattern_by_row[r] = []
            pattern_by_row[r].append(c)
        
        # Test hypothesis: For each row with pattern, reflect pattern across center
        predicted_changes = set()
        
        for r in range(min_r, max_r + 1):
            if r in pattern_by_row:
                pattern_cols = sorted(pattern_by_row[r])
                left_most = min(pattern_cols)
                right_most = max(pattern_cols)
                
                # Calculate center for this row's pattern
                row_center = (left_most + right_most) / 2.0
                
                print(f"\nRow {r}: pattern cols {pattern_cols}")
                print(f"  Leftmost: {left_most}, Rightmost: {right_most}, Center: {row_center}")
                
                # For each pattern cell in this row, find its reflection
                for pc in pattern_cols:
                    reflected_c = int(round(2 * row_center - pc))
                    
                    # If reflection would land on background within reasonable bounds
                    if (min_c <= reflected_c <= max_c and 
                        input_grid[r][reflected_c] == bg_color and
                        (r, reflected_c) not in pattern_cells):
                        predicted_changes.add((r, reflected_c))
                        print(f"  Pattern at ({r},{pc}) → reflect to ({r},{reflected_c})")
        
        correct_predictions = predicted_changes & changed_cells
        missed = changed_cells - predicted_changes
        extra = predicted_changes - changed_cells
        
        print(f"\nPrediction results:")
        print(f"  Correct: {len(correct_predictions)}/{len(changed_cells)}")
        print(f"  Missed: {missed}")
        print(f"  Extra: {extra}")
        
        if len(predicted_changes) > 0:
            precision = len(correct_predictions) / len(predicted_changes)
            recall = len(correct_predictions) / len(changed_cells) if len(changed_cells) > 0 else 1.0
            print(f"  Precision: {precision:.2%}")
            print(f"  Recall: {recall:.2%}")

if __name__ == "__main__":
    test_opposite_side_hypothesis()