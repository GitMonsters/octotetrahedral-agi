#!/usr/bin/env python3

import json
import numpy as np

def analyze_all_patterns():
    with open('/Users/evanpieser/apr12_tasks/651b2ee5.json', 'r') as f:
        task = json.load(f)
    
    print("=== ANALYZING PATTERNS FOR ALL TRAINING PAIRS ===\n")
    
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
        
        print(f"  Marker color: {marker_color}")
        
        h, w = output_grid.shape
        
        # Test different patterns
        patterns = [4, 8]
        
        for pattern in patterns:
            # Test diagonal pattern
            predicted = np.ones((h, w), dtype=int)
            for y in range(h):
                for x in range(w):
                    if (y + x) % pattern == 0 or (y - x) % pattern == 0:
                        predicted[y, x] = marker_color
            
            differences = np.sum(predicted != output_grid)
            total = h * w
            accuracy = (total - differences) / total
            
            print(f"  Pattern {pattern}-diagonal: {accuracy:.3f} accuracy ({differences} differences)")
        
        # Also try different modulo values for main diagonal only
        for mod in [2, 3, 4, 5, 6, 8]:
            predicted = np.ones((h, w), dtype=int)
            for y in range(h):
                for x in range(w):
                    if (y + x) % mod == 0:
                        predicted[y, x] = marker_color
            
            differences = np.sum(predicted != output_grid)
            accuracy = (total - differences) / total
            if accuracy > 0.9:
                print(f"  (y+x) % {mod}: {accuracy:.3f} accuracy")
        
        # Try anti-diagonal only
        for mod in [2, 3, 4, 5, 6, 8]:
            predicted = np.ones((h, w), dtype=int)
            for y in range(h):
                for x in range(w):
                    if (y - x) % mod == 0:
                        predicted[y, x] = marker_color
            
            differences = np.sum(predicted != output_grid)
            accuracy = (total - differences) / total
            if accuracy > 0.9:
                print(f"  (y-x) % {mod}: {accuracy:.3f} accuracy")
        
        print()

if __name__ == "__main__":
    analyze_all_patterns()