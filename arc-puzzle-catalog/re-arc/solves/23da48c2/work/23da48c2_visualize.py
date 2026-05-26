#!/usr/bin/env python3

import json
import numpy as np
from collections import Counter

def visualize_transformation():
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        data = json.load(f)
    
    for pair_idx in range(len(data['train'])):
        pair = data['train'][pair_idx]
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"\n=== Training Pair {pair_idx + 1} ===")
        
        # Find background color
        input_flat = input_grid.flatten()
        background_color = Counter(input_flat).most_common(1)[0][0]
        
        # Print input grid with coordinates (small portion)
        print("Input grid (top-left 10x10):")
        print("   ", end="")
        for c in range(min(10, input_grid.shape[1])):
            print(f"{c:2}", end="")
        print()
        
        for r in range(min(10, input_grid.shape[0])):
            print(f"{r:2} ", end="")
            for c in range(min(10, input_grid.shape[1])):
                val = input_grid[r, c]
                if val == background_color:
                    print(" .", end="")
                else:
                    print(f"{val:2}", end="")
            print()
        
        print(f"\nOutput grid (top-left 10x{min(10, output_grid.shape[1])}):")
        print("   ", end="")
        for c in range(min(10, output_grid.shape[1])):
            print(f"{c:2}", end="")
        print()
        
        for r in range(min(10, output_grid.shape[0])):
            print(f"{r:2} ", end="")
            for c in range(min(10, output_grid.shape[1])):
                if r < output_grid.shape[0] and c < output_grid.shape[1]:
                    val = output_grid[r, c]
                    if val == background_color:
                        print(" .", end="")
                    else:
                        print(f"{val:2}", end="")
                else:
                    print("  ", end="")
            print()
        
        if pair_idx == 0:  # Just show first pair in detail for now
            break

if __name__ == "__main__":
    visualize_transformation()