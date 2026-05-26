#!/usr/bin/env python3

import json
import numpy as np

def test_cyclic_pattern():
    with open('/Users/evanpieser/apr12_tasks/651b2ee5.json', 'r') as f:
        task = json.load(f)
    
    pair = task['train'][2]
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    print("=== TESTING CYCLIC PATTERN FOR PAIR 3 ===")
    
    h, w = output_grid.shape
    
    # Define the observed pattern for first 8 rows
    pattern_by_row = {
        0: [2, 4],
        1: [1, 3],
        2: [0, 2],
        3: [1],
        4: [0, 2],
        5: [1, 3],
        6: [2, 4],
        7: [3],      # From the observed data
    }
    
    # Let's see if this pattern continues
    print("Actual vs Pattern-based prediction:")
    
    predicted = np.ones((h, w), dtype=int)
    
    for y in range(h):
        # Use modulo to repeat the pattern
        pattern_row = y % 8  # Try 8-row cycle
        
        # Get the marker positions for this pattern row
        if pattern_row in pattern_by_row:
            marker_cols = pattern_by_row[pattern_row]
            for x in marker_cols:
                if x < w:  # Make sure we don't go out of bounds
                    predicted[y, x] = 6
    
    # Compare
    differences = np.sum(predicted != output_grid)
    accuracy = (h * w - differences) / (h * w)
    print(f"8-row cycle: {differences} errors, {accuracy:.3f} accuracy")
    
    # Show differences
    if differences > 0:
        print("\nActual vs Predicted (first 10 rows):")
        for y in range(min(10, h)):
            actual_row = [output_grid[y, x] for x in range(w)]
            predicted_row = [predicted[y, x] for x in range(w)]
            
            actual_markers = [x for x in range(w) if actual_row[x] == 6]
            predicted_markers = [x for x in range(w) if predicted_row[x] == 6]
            
            match = "✓" if actual_markers == predicted_markers else "✗"
            print(f"Row {y:2d}: actual={actual_markers}, predicted={predicted_markers} {match}")
    
    # Let's try to figure out the actual pattern by looking at more rows
    print(f"\nAll actual marker positions by row:")
    for y in range(h):
        markers = [x for x in range(w) if output_grid[y, x] == 6]
        print(f"Row {y:2d}: {markers}")
    
    # Look for a mathematical pattern
    print(f"\nLooking for mathematical pattern:")
    for y in range(h):
        markers = [x for x in range(w) if output_grid[y, x] == 6]
        for x in markers:
            print(f"  ({y},{x}): y%7={y%7}, x%5={x%5}, (y+x)%2={(y+x)%2}, (y+x)%3={(y+x)%3}")

if __name__ == "__main__":
    test_cyclic_pattern()