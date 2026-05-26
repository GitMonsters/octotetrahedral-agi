import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load task data
with open('/Users/evanpieser/apr12_tasks/6b7e1999.json', 'r') as f:
    data = json.load(f)

# ARC color palette
colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
          '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']

def plot_grid(grid, title):
    grid = np.array(grid)
    plt.imshow(grid, cmap=ListedColormap(colors[:10]), vmin=0, vmax=9)
    plt.title(title, fontsize=10)
    plt.axis('off')
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            plt.text(j, i, str(grid[i,j]), ha='center', va='center', fontsize=6)

# Create visualization
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
fig.suptitle('ARC-AGI Task 6b7e1999 - Object-based Analysis', fontsize=16)

# Train examples
for i, example in enumerate(data['train']):
    plt.subplot(2, 6, i*2 + 1)
    plot_grid(example['input'], f'Train {i} Input')
    
    plt.subplot(2, 6, i*2 + 2)
    plot_grid(example['output'], f'Train {i} Output')

plt.tight_layout()
plt.savefig('/Users/evanpieser/6b7e1999_obj_viz.png', dpi=150, bbox_inches='tight')
plt.show()

# Now let's analyze differences
print("=== CELL-BY-CELL DIFFERENCES ===")
for i, example in enumerate(data['train']):
    input_grid = example['input']
    output_grid = example['output']
    print(f"\nTrain pair {i}:")
    changes = []
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r][c] != output_grid[r][c]:
                changes.append((r, c, input_grid[r][c], output_grid[r][c]))
    
    print(f"  {len(changes)} cells changed:")
    for change in changes:
        r, c, old_val, new_val = change
        print(f"    ({r},{c}): {old_val} -> {new_val}")