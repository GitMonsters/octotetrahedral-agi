import json
import matplotlib.pyplot as plt
import numpy as np

# Load task data
with open('/Users/evanpieser/apr12_tasks/76d965ef.json', 'r') as f:
    data = json.load(f)

# Color mapping for visualization
colors = {
    0: '#000000',  # black
    1: '#0074D9',  # blue
    2: '#FF4136',  # red  
    3: '#2ECC40',  # green (background)
    4: '#FFDC00',  # yellow
    5: '#FF851B',  # orange
    6: '#B10DC9',  # purple
    7: '#85144b',  # maroon
    8: '#F012BE',  # magenta
    9: '#FFFFFF'   # white
}

def grid_to_image(grid):
    """Convert grid to RGB array for visualization"""
    h, w = len(grid), len(grid[0])
    img = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            color = colors[grid[i][j]]
            img[i, j] = [int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255]
    return img

# Create visualization
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
fig.suptitle('ARC-AGI Task 76d965ef: Input → Output Pattern Analysis', fontsize=16)

for i, example in enumerate(data['train']):
    # Input
    input_grid = example['input']
    input_img = grid_to_image(input_grid)
    axes[i, 0].imshow(input_img)
    axes[i, 0].set_title(f'Train {i} Input: {len(input_grid)}x{len(input_grid[0])}')
    axes[i, 0].set_xticks(range(len(input_grid[0])))
    axes[i, 0].set_yticks(range(len(input_grid)))
    axes[i, 0].grid(True, alpha=0.3)
    
    # Output  
    output_grid = example['output']
    output_img = grid_to_image(output_grid)
    axes[i, 1].imshow(output_img)
    axes[i, 1].set_title(f'Train {i} Output: {len(output_grid)}x{len(output_grid[0])}')
    axes[i, 1].set_xticks(range(0, len(output_grid[0]), 2))
    axes[i, 1].set_yticks(range(0, len(output_grid), 2))
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/evanpieser/76d965ef_viz.png', dpi=150, bbox_inches='tight')
print("Visualization saved to 76d965ef_viz.png")

# Print detailed analysis
print("\n=== DETAILED PATTERN ANALYSIS ===")
for i, example in enumerate(data['train']):
    input_grid = example['input']
    output_grid = example['output']
    
    print(f"\nTrain {i}:")
    print(f"  Input: {len(input_grid)}x{len(input_grid[0])} → Output: {len(output_grid)}x{len(output_grid[0])}")
    
    # Find non-background pattern
    min_r = max_r = min_c = max_c = None
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r][c] != 3:
                if min_r is None:
                    min_r = max_r = r
                    min_c = max_c = c
                else:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
    
    if min_r is not None:
        pattern_h = max_r - min_r + 1
        pattern_w = max_c - min_c + 1
        print(f"  Pattern region: ({min_r},{min_c}) to ({max_r},{max_c}) = {pattern_h}x{pattern_w}")
        
        # Extract pattern
        pattern = []
        for r in range(min_r, max_r + 1):
            row = [input_grid[r][c] for c in range(min_c, max_c + 1)]
            pattern.append(row)
            
        print(f"  Pattern:")
        for row in pattern:
            print(f"    {row}")