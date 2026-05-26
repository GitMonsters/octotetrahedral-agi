#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def load_task():
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        return json.load(f)

def create_colormap():
    """Create colormap for ARC colors 0-9"""
    colors = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'magenta', 'orange', 'lightblue', 'brown']
    return mcolors.ListedColormap(colors)

def visualize_task(task):
    """Create visualization of all training pairs"""
    train_pairs = task['train']
    n_pairs = len(train_pairs)
    
    fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 3*n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    cmap = create_colormap()
    
    for i, pair in enumerate(train_pairs):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        # Plot input
        axes[i, 0].imshow(input_grid, cmap=cmap, vmin=0, vmax=9)
        axes[i, 0].set_title(f'Train {i} Input ({input_grid.shape[0]}x{input_grid.shape[1]})')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot output  
        axes[i, 1].imshow(output_grid, cmap=cmap, vmin=0, vmax=9)
        axes[i, 1].set_title(f'Train {i} Output ({output_grid.shape[0]}x{output_grid.shape[1]})')
        axes[i, 1].grid(True, alpha=0.3)
        
        print(f"Train {i}: {input_grid.shape[0]}x{input_grid.shape[1]} -> {output_grid.shape[0]}x{output_grid.shape[1]}")
        print(f"  Input cols removed: {input_grid.shape[1] - output_grid.shape[1]}")
    
    plt.tight_layout()
    plt.savefig('/Users/evanpieser/23da48c2_viz.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    task = load_task()
    print("TASK 23da48c2 ANALYSIS")
    print("=" * 40)
    visualize_task(task)