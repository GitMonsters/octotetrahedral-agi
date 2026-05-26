#!/usr/bin/env python3
"""
Visualize ARC task 32616820 - all train pairs
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ARC color palette
arc_colors = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: gray
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: cyan
    '#870C25'   # 9: brown
]

def visualize_grid(grid, title):
    grid = np.array(grid)
    fig, ax = plt.subplots(figsize=(max(8, grid.shape[1]*0.4), max(6, grid.shape[0]*0.4)))
    
    # Create custom colormap
    cmap = ListedColormap(arc_colors[:10])
    
    # Plot grid
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return fig, ax

def main():
    # Load task
    with open('/Users/evanpieser/apr12_tasks/32616820.json', 'r') as f:
        task = json.load(f)
    
    # Create visualizations for all training pairs
    for i, pair in enumerate(task['train']):
        input_grid = pair['input']
        output_grid = pair['output']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot input
        input_arr = np.array(input_grid)
        cmap = ListedColormap(arc_colors[:10])
        im1 = ax1.imshow(input_arr, cmap=cmap, vmin=0, vmax=9)
        
        # Add grid lines
        ax1.set_xticks(np.arange(-0.5, input_arr.shape[1], 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, input_arr.shape[0], 1), minor=True)
        ax1.grid(which="minor", color="white", linestyle='-', linewidth=1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f'Training Pair {i+1} - INPUT ({input_arr.shape[0]}x{input_arr.shape[1]})', fontweight='bold')
        
        # Plot output  
        output_arr = np.array(output_grid)
        im2 = ax2.imshow(output_arr, cmap=cmap, vmin=0, vmax=9)
        
        # Add grid lines
        ax2.set_xticks(np.arange(-0.5, output_arr.shape[1], 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, output_arr.shape[0], 1), minor=True)
        ax2.grid(which="minor", color="white", linestyle='-', linewidth=1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title(f'Training Pair {i+1} - OUTPUT ({output_arr.shape[0]}x{output_arr.shape[1]})', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'/Users/evanpieser/32616820_train_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for training pair {i+1}")
        
        # Print some analysis
        print(f"Training pair {i+1}:")
        print(f"  Input shape: {input_arr.shape}")
        print(f"  Output shape: {output_arr.shape}")
        
        # Count unique colors
        input_colors = set(input_arr.flatten())
        output_colors = set(output_arr.flatten())
        print(f"  Input colors: {sorted(input_colors)}")
        print(f"  Output colors: {sorted(output_colors)}")
        print()

if __name__ == '__main__':
    main()