#!/usr/bin/env python3
"""
Visualize ARC task 4e600a86 to understand the transformation pattern
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_task(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def visualize_grid(grid, title, ax):
    """Visualize a grid with proper colors"""
    grid = np.array(grid)
    h, w = grid.shape
    
    # Color mapping for ARC
    colors = {
        0: '#000000',  # black
        1: '#0074D9',  # blue  
        2: '#FF4136',  # red
        3: '#2ECC40',  # green
        4: '#FFDC00',  # yellow
        5: '#AAAAAA',  # gray
        6: '#F012BE',  # magenta
        7: '#FF851B',  # orange
        8: '#7FDBFF',  # aqua
        9: '#870C25'   # maroon
    }
    
    # Create colored image
    img = np.zeros((h, w, 3))
    for value, color in colors.items():
        mask = (grid == value)
        color_rgb = [int(color[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
        img[mask] = color_rgb
    
    ax.imshow(img, aspect='equal')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid lines
    for i in range(h + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)
    for j in range(w + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5)

def analyze_differences(input_grid, output_grid):
    """Analyze cell-by-cell differences between input and output"""
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)
    
    changes = []
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_arr[r, c] != output_arr[r, c]:
                changes.append(f"({r}, {c}): {input_arr[r, c]} -> {output_arr[r, c]}")
    
    return changes

def main():
    # Load the task
    task = load_task('/Users/evanpieser/apr12_tasks/4e600a86.json')
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Visualize all training pairs
    for i, pair in enumerate(task['train']):
        # Input
        ax_in = fig.add_subplot(3, 4, i*4 + 1)
        visualize_grid(pair['input'], f'Train {i+1} Input', ax_in)
        
        # Output  
        ax_out = fig.add_subplot(3, 4, i*4 + 2)
        visualize_grid(pair['output'], f'Train {i+1} Output', ax_out)
        
        # Differences
        ax_diff = fig.add_subplot(3, 4, i*4 + 3)
        input_arr = np.array(pair['input'])
        output_arr = np.array(pair['output'])
        diff = (input_arr != output_arr).astype(int)
        ax_diff.imshow(diff, cmap='Reds', aspect='equal')
        ax_diff.set_title(f'Train {i+1} Changes', fontsize=10)
        ax_diff.set_xticks([])
        ax_diff.set_yticks([])
        
        # Print analysis
        changes = analyze_differences(pair['input'], pair['output'])
        print(f"\n=== TRAIN PAIR {i+1} ===")
        print(f"Input size: {len(pair['input'])} x {len(pair['input'][0])}")
        print(f"Output size: {len(pair['output'])} x {len(pair['output'][0])}")
        print(f"Number of changes: {len(changes)}")
        if changes:
            print("Changes (row, col): old -> new")
            for change in changes[:10]:  # Show first 10 changes
                print(f"  {change}")
            if len(changes) > 10:
                print(f"  ... and {len(changes) - 10} more")
    
    # Visualize test cases  
    for i, test_case in enumerate(task['test']):
        ax_test = fig.add_subplot(3, 4, 9 + i)
        visualize_grid(test_case['input'], f'Test {i+1} Input', ax_test)
    
    plt.tight_layout()
    plt.savefig('/Users/evanpieser/4e600a86_viz.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: 4e600a86_viz.png")
    plt.show()

if __name__ == '__main__':
    main()