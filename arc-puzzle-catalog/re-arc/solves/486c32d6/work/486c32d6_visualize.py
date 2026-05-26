import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Load the task data
with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
    task_data = json.load(f)

# Define colors for visualization
colors = ['#000000', '#FFFFFF', '#FF0000', '#00FF00', '#0000FF', 
          '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080']
cmap = ListedColormap(colors)

# Function to visualize grid pairs
def visualize_pair(input_grid, output_grid, title):
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Input grid
    ax1.imshow(input_arr, cmap=cmap, vmin=0, vmax=9)
    ax1.set_title(f'{title} - Input')
    ax1.set_xticks(range(len(input_grid[0])))
    ax1.set_yticks(range(len(input_grid)))
    ax1.grid(True, color='gray', linewidth=0.5)
    
    # Output grid  
    ax2.imshow(output_arr, cmap=cmap, vmin=0, vmax=9)
    ax2.set_title(f'{title} - Output')
    ax2.set_xticks(range(len(output_grid[0])))
    ax2.set_yticks(range(len(output_grid)))
    ax2.grid(True, color='gray', linewidth=0.5)
    
    return fig

# Create visualizations for all training examples
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for i, example in enumerate(task_data['train']):
    input_grid = example['input']
    output_grid = example['output']
    
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)
    
    # Input grid
    axes[i,0].imshow(input_arr, cmap=cmap, vmin=0, vmax=9)
    axes[i,0].set_title(f'Train {i} - Input ({len(input_grid)}x{len(input_grid[0])})')
    axes[i,0].set_xticks(range(len(input_grid[0])))
    axes[i,0].set_yticks(range(len(input_grid)))
    axes[i,0].grid(True, color='gray', linewidth=0.5)
    
    # Output grid  
    axes[i,1].imshow(output_arr, cmap=cmap, vmin=0, vmax=9)
    axes[i,1].set_title(f'Train {i} - Output ({len(output_grid)}x{len(output_grid[0])})')
    axes[i,1].set_xticks(range(len(output_grid[0])))
    axes[i,1].set_yticks(range(len(output_grid)))
    axes[i,1].grid(True, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig('/Users/evanpieser/486c32d6_viz.png', dpi=150, bbox_inches='tight')
plt.show()

# Analyze differences
print("=== ANALYSIS ===")
for i, example in enumerate(task_data['train']):
    print(f"\nTrain {i}:")
    input_grid = example['input']
    output_grid = example['output']
    
    # Find differences
    differences = []
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r][c] != output_grid[r][c]:
                differences.append(f"  Row {r}, Col {c}: {input_grid[r][c]} → {output_grid[r][c]}")
    
    if differences:
        print(f"  Differences found:")
        for diff in differences[:10]:  # Show first 10 differences
            print(diff)
        if len(differences) > 10:
            print(f"  ... and {len(differences) - 10} more")
    else:
        print("  No differences found")