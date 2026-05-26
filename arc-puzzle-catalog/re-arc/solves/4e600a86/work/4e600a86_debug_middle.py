#!/usr/bin/env python3
import json

# Load task data
with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
    task = json.load(f)

def debug_exact_middle_hypothesis():
    """Debug the exact middle hypothesis"""
    train_pairs = task['train']
    
    for i, pair in enumerate(train_pairs):
        print(f"\n{'='*60}")
        print(f"EXACT MIDDLE HYPOTHESIS DEBUG - TRAIN PAIR {i+1}")
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
        
        center_c = (min_c + max_c) / 2.0  # Exact middle
        
        print(f"Pattern bbox: ({min_r},{min_c}) to ({max_r},{max_c})")
        print(f"Exact center column: {center_c}")
        
        # Find actual changes
        actual_changes = set()
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != output_grid[r][c]:
                    actual_changes.add((r, c))
        
        print("\nDetailed analysis of each background cell in bbox:")
        
        correct_predictions = 0
        total_predictions = 0
        
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if input_grid[r][c] == bg_color:  # Background cell
                    reflected_c = int(2 * center_c - c)
                    has_pattern_at_reflection = (min_c <= reflected_c <= max_c and 
                                               (r, reflected_c) in pattern_cells)
                    actually_changed = (r, c) in actual_changes
                    
                    should_predict = has_pattern_at_reflection
                    
                    if should_predict:
                        total_predictions += 1
                        if actually_changed:
                            correct_predictions += 1
                            status = "✓ CORRECT"
                        else:
                            status = "✗ FALSE POSITIVE"
                    else:
                        if actually_changed:
                            status = "✗ MISSED"
                        else:
                            status = "- not predicted"
                            continue  # Skip printing for uninteresting cases
                    
                    print(f"  ({r},{c}) bg → reflect to c={reflected_c}, "
                          f"has_pattern={has_pattern_at_reflection}, "
                          f"changed={actually_changed} → {status}")
        
        print(f"\nCorrect predictions: {correct_predictions}/{total_predictions}")
        
        # Let me also try a different approach - maybe it's not reflection but completion
        print(f"\nTrying pattern completion hypothesis...")
        
        # For each background cell, check if filling it creates symmetry
        symmetry_predictions = 0
        symmetry_correct = 0
        
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if input_grid[r][c] == bg_color:
                    # Check if this position would create horizontal symmetry
                    # Test if there's a pattern cell at the mirrored position
                    mirrored_c = int(2 * center_c - c)
                    
                    if min_c <= mirrored_c <= max_c:
                        has_mirror = (r, mirrored_c) in pattern_cells
                        if has_mirror:
                            symmetry_predictions += 1
                            if (r, c) in actual_changes:
                                symmetry_correct += 1

def test_center_line_hypothesis():
    """Test if it's about reflecting across the center LINE not center POINT"""
    train_pairs = task['train']
    
    print(f"\n{'='*60}")
    print("TESTING CENTER LINE HYPOTHESIS")
    print(f"{'='*60}")
    
    for i, pair in enumerate(train_pairs):
        print(f"\n--- TRAIN PAIR {i+1} ---")
        
        input_grid = pair['input']
        output_grid = pair['output']
        h, w = len(input_grid), len(input_grid[0])
        
        # Find background and pattern
        colors = {}
        for row in input_grid:
            for cell in row:
                colors[cell] = colors.get(cell, 0) + 1
        bg_color = max(colors, key=colors.get)
        
        if bg_color == 3:
            continue
        
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
        
        actual_changes = set()
        for r in range(h):
            for c in range(w):
                if input_grid[r][c] != output_grid[r][c]:
                    actual_changes.add((r, c))
        
        # Test reflection across center line between columns
        center_line = (min_c + max_c) / 2.0  # This is between two columns if width is even
        
        correct = 0
        total_pred = 0
        
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if input_grid[r][c] == bg_color:
                    # Reflect across the center line
                    reflected_c = 2 * center_line - c
                    reflected_c_int = int(round(reflected_c))
                    
                    if (min_c <= reflected_c_int <= max_c and 
                        (r, reflected_c_int) in pattern_cells):
                        total_pred += 1
                        if (r, c) in actual_changes:
                            correct += 1
        
        if total_pred > 0:
            precision = correct / total_pred
            recall = correct / len(actual_changes) if len(actual_changes) > 0 else 1.0
            print(f"Center line reflection: precision={precision:.2%}, recall={recall:.2%}")

if __name__ == "__main__":
    debug_exact_middle_hypothesis()
    test_center_line_hypothesis()