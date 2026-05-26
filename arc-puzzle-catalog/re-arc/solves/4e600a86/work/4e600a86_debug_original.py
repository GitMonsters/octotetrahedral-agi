#!/usr/bin/env python3
import json

# Load task data
with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
    task = json.load(f)

# Load and test original solver
exec(open('/Users/evanpieser/apr12_solvers/4e600a86_solver.py').read())

def test_original_solver_in_detail():
    """Test the original solver and debug its logic"""
    train_pairs = task['train']
    
    for i, pair in enumerate(train_pairs):
        print(f"\n{'='*60}")
        print(f"TESTING ORIGINAL SOLVER - TRAIN PAIR {i+1}")
        print(f"{'='*60}")
        
        input_grid = pair['input']
        expected_output = pair['output']
        
        # Test original solver
        predicted_output = solve(input_grid)
        
        # Manual verification of the vertical reflection logic
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
        
        if not pattern_cells:
            continue
        
        min_r = min(r for r, c in pattern_cells)
        max_r = max(r for r, c in pattern_cells)
        min_c = min(c for r, c in pattern_cells)
        max_c = max(c for r, c in pattern_cells)
        
        pattern_height = max_r - min_r + 1
        center_idx = pattern_height // 2
        center_r = min_r + center_idx
        
        print(f"Pattern bbox: ({min_r},{min_c}) to ({max_r},{max_c})")
        print(f"Pattern height: {pattern_height}, center_idx: {center_idx}, center_r: {center_r}")
        
        # Debug the vertical reflection logic
        predicted_changes_debug = []
        
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if input_grid[r][c] == bg_color:
                    dist_from_center = r - center_r
                    reflected_r = center_r - dist_from_center
                    
                    if (min_r <= reflected_r <= max_r and 
                        (reflected_r, c) in pattern_cells):
                        predicted_changes_debug.append((r, c, reflected_r))
        
        # Find actual changes
        actual_changes = set()
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != expected_output[r][c]:
                    actual_changes.add((r, c))
        
        predicted_changes_set = set((r, c) for r, c, _ in predicted_changes_debug)
        
        print(f"Predicted changes: {len(predicted_changes_set)}")
        print(f"Actual changes: {len(actual_changes)}")
        
        correct = predicted_changes_set & actual_changes
        missed = actual_changes - predicted_changes_set
        extra = predicted_changes_set - actual_changes
        
        print(f"Correct: {len(correct)}")
        print(f"Missed: {len(missed)} - {list(missed)[:5]}")
        print(f"Extra: {len(extra)} - {list(extra)[:5]}")
        
        if len(predicted_changes_debug) > 0:
            print(f"\nFirst few predicted changes (debugging):")
            for r, c, reflected_r in predicted_changes_debug[:10]:
                in_actual = (r, c) in actual_changes
                print(f"  ({r},{c}) ← reflects from ({reflected_r},{c}): {'✓' if in_actual else '✗'}")

if __name__ == "__main__":
    test_original_solver_in_detail()