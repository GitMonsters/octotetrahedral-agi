#!/usr/bin/env python3

import json
import numpy as np

def test_aspect_ratio_pattern():
    with open('/Users/evanpieser/apr12_tasks/651b2ee5.json', 'r') as f:
        task = json.load(f)
    
    print("=== TESTING ASPECT RATIO PATTERN HYPOTHESIS ===\n")
    
    for i, pair in enumerate(task['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"TRAIN PAIR {i+1}: {input_grid.shape}")
        
        # Find marker colors
        unique_colors = np.unique(input_grid)
        color_counts = {}
        for color in unique_colors:
            color_counts[color] = np.sum(input_grid == color)
        
        background_color = max(color_counts, key=color_counts.get)
        marker_colors = [c for c in unique_colors if c != background_color]
        marker_color = marker_colors[0]
        
        h, w = output_grid.shape
        aspect_ratio = max(h, w) / min(h, w)
        
        print(f"  Aspect ratio: {aspect_ratio:.2f} ({h}x{w})")
        
        # Try different patterns based on aspect ratio
        if aspect_ratio >= 3.0:
            # Very rectangular: use checkerboard
            print(f"  Using checkerboard pattern (high aspect ratio)")
            predicted = np.ones((h, w), dtype=int)
            for y in range(h):
                for x in range(w):
                    if (y + x) % 2 == 0:
                        predicted[y, x] = marker_color
        
        elif aspect_ratio >= 1.8:
            # Moderately rectangular: use 8-diagonal for wide, 4-diagonal for tall
            if w > h:
                print(f"  Using 8-diagonal pattern (wide rectangle)")
                predicted = np.ones((h, w), dtype=int)
                for y in range(h):
                    for x in range(w):
                        if (y + x) % 8 == 0 or (y - x) % 8 == 0:
                            predicted[y, x] = marker_color
            else:
                print(f"  Using 4-diagonal pattern (tall rectangle)")
                predicted = np.ones((h, w), dtype=int)
                for y in range(h):
                    for x in range(w):
                        if (y + x) % 4 == 0 or (y - x) % 4 == 0:
                            predicted[y, x] = marker_color
        
        else:
            # Nearly square: use 4-diagonal
            print(f"  Using 4-diagonal pattern (nearly square)")
            predicted = np.ones((h, w), dtype=int)
            for y in range(h):
                for x in range(w):
                    if (y + x) % 4 == 0 or (y - x) % 4 == 0:
                        predicted[y, x] = marker_color
        
        differences = np.sum(predicted != output_grid)
        accuracy = (h * w - differences) / (h * w)
        
        print(f"  Accuracy: {accuracy:.3f} ({differences} differences)")
        
        if differences == 0:
            print(f"  ✓ PERFECT MATCH!")
        else:
            print(f"  ✗ Still has errors")
        
        print()

if __name__ == "__main__":
    test_aspect_ratio_pattern()