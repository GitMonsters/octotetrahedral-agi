#!/usr/bin/env python3

import json
import numpy as np
from collections import Counter

def analyze_repositioning():
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        data = json.load(f)
    
    # Focus on one pair first to understand the pattern
    pair = data['train'][0]
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    print("=== Training Pair 1 - Detailed Analysis ===")
    print(f"Input shape: {input_grid.shape}")
    print(f"Output shape: {output_grid.shape}")
    
    # Find background color
    input_flat = input_grid.flatten()
    background_color = Counter(input_flat).most_common(1)[0][0]
    print(f"Background color: {background_color}")
    
    # Find all non-background pixels in input
    input_non_bg = np.where(input_grid != background_color)
    input_coords = list(zip(input_non_bg[0], input_non_bg[1]))
    input_values = [input_grid[r, c] for r, c in input_coords]
    
    print(f"Non-background pixels in input: {len(input_coords)}")
    
    # Find all non-background pixels in output  
    output_non_bg = np.where(output_grid != background_color)
    output_coords = list(zip(output_non_bg[0], output_non_bg[1]))
    output_values = [output_grid[r, c] for r, c in output_coords]
    
    print(f"Non-background pixels in output: {len(output_coords)}")
    
    # Compare values
    input_color_counts = Counter(input_values)
    output_color_counts = Counter(output_values)
    
    print(f"Input color counts: {dict(input_color_counts)}")
    print(f"Output color counts: {dict(output_color_counts)}")
    
    print(f"Color counts match: {input_color_counts == output_color_counts}")
    
    # Look at specific positions
    print("\nFirst few input non-bg positions and values:")
    for i in range(min(10, len(input_coords))):
        r, c = input_coords[i]
        print(f"  Input[{r},{c}] = {input_grid[r,c]}")
    
    print("\nFirst few output non-bg positions and values:")
    for i in range(min(10, len(output_coords))):
        r, c = output_coords[i]
        print(f"  Output[{r},{c}] = {output_grid[r,c]}")

if __name__ == "__main__":
    analyze_repositioning()