#!/usr/bin/env python3
import json

# Load task data
with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
    task = json.load(f)

def test_horizontal_reflection_hypothesis():
    """Test the horizontal reflection hypothesis precisely"""
    train_pairs = task['train']
    
    for i, pair in enumerate(train_pairs):
        print(f"\n{'='*50}")
        print(f"TESTING HYPOTHESIS - TRAIN PAIR {i+1}")
        print(f"{'='*50}")
        
        input_grid = pair['input']
        output_grid = pair['output']
        h, w = len(input_grid), len(input_grid[0])
        
        # Find background color
        colors = {}
        for row in input_grid:
            for cell in row:
                colors[cell] = colors.get(cell, 0) + 1
        bg_color = max(colors, key=colors.get)
        
        # Skip if background is already 3
        if bg_color == 3:
            print("Background is already 3 - no transformation needed")
            continue
        
        # Find pattern cells
        pattern_cells = set()
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != bg_color:
                    pattern_cells.add((r, c))
        
        if not pattern_cells:
            continue
            
        # Pattern bounding box
        min_r = min(r for r, c in pattern_cells)
        max_r = max(r for r, c in pattern_cells)
        min_c = min(c for r, c in pattern_cells)
        max_c = max(c for r, c in pattern_cells)
        
        print(f"Background: {bg_color}")
        print(f"Pattern bbox: ({min_r},{min_c}) to ({max_r},{max_c})")
        
        # Test horizontal reflection hypothesis
        pattern_width = max_c - min_c + 1
        center_c = min_c + pattern_width // 2
        
        print(f"Pattern width: {pattern_width}, center column: {center_c}")
        
        predicted_changes = []
        
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if input_grid[r][c] == bg_color:  # This is a background cell
                    # Find horizontally reflected position
                    reflected_c = 2 * center_c - c
                    
                    if min_c <= reflected_c <= max_c and (r, reflected_c) in pattern_cells:
                        predicted_changes.append((r, c))
        
        # Find actual changes
        actual_changes = []
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != output_grid[r][c]:
                    actual_changes.append((r, c))
        
        predicted_set = set(predicted_changes)
        actual_set = set(actual_changes)
        
        correct_predictions = predicted_set & actual_set
        missed_changes = actual_set - predicted_set
        false_positives = predicted_set - actual_set
        
        print(f"\nPredicted changes: {len(predicted_changes)}")
        print(f"Actual changes: {len(actual_changes)}")
        print(f"Correct predictions: {len(correct_predictions)}")
        print(f"Missed changes: {len(missed_changes)}")
        print(f"False positives: {len(false_positives)}")
        
        if len(predicted_changes) > 0:
            precision = len(correct_predictions) / len(predicted_changes)
            recall = len(correct_predictions) / len(actual_changes) if len(actual_changes) > 0 else 1.0
            print(f"Precision: {precision:.2%}")
            print(f"Recall: {recall:.2%}")
            
        if missed_changes:
            print(f"Missed changes: {missed_changes}")
        if false_positives:
            print(f"False positives: {false_positives}")

def test_alternative_hypotheses():
    """Test some alternative hypotheses"""
    print("\n" + "="*60)
    print("TESTING ALTERNATIVE HYPOTHESES")
    print("="*60)
    
    # Let me try a simpler hypothesis: mirror the pattern across the vertical centerline
    train_pairs = task['train']
    
    for i, pair in enumerate(train_pairs):
        print(f"\n--- TRAIN PAIR {i+1} ---")
        
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
            continue
            
        # Find pattern cells
        pattern_cells = set()
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != bg_color:
                    pattern_cells.add((r, c))
        
        if not pattern_cells:
            continue
            
        # Pattern bounding box
        min_r = min(r for r, c in pattern_cells)
        max_r = max(r for r, c in pattern_cells)
        min_c = min(c for r, c in pattern_cells)
        max_c = max(c for r, c in pattern_cells)
        
        # Try different center calculations
        centers_to_try = [
            ("floor", min_c + (max_c - min_c) // 2),
            ("ceil", min_c + (max_c - min_c + 1) // 2),
            ("exact_middle", (min_c + max_c) / 2.0)
        ]
        
        # Find actual changes
        actual_changes = set()
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != output_grid[r][c]:
                    actual_changes.add((r, c))
        
        for center_name, center_c in centers_to_try:
            predicted_changes = set()
            
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if input_grid[r][c] == bg_color:  # Background cell
                        if isinstance(center_c, float):
                            # For exact middle, use different reflection formula
                            reflected_c = int(2 * center_c - c)
                        else:
                            reflected_c = 2 * center_c - c
                        
                        if min_c <= reflected_c <= max_c and (r, reflected_c) in pattern_cells:
                            predicted_changes.add((r, c))
            
            correct = len(predicted_changes & actual_changes)
            total_predicted = len(predicted_changes)
            total_actual = len(actual_changes)
            
            if total_predicted > 0:
                precision = correct / total_predicted
                recall = correct / total_actual if total_actual > 0 else 1.0
                print(f"  {center_name} center={center_c}: precision={precision:.2%}, recall={recall:.2%}")

if __name__ == "__main__":
    test_horizontal_reflection_hypothesis()
    test_alternative_hypotheses()