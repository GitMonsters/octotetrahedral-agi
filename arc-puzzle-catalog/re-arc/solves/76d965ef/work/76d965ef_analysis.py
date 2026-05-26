#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data
with open('/Users/evanpieser/apr12_tasks/76d965ef.json', 'r') as f:
    data = json.load(f)

def visualize_task():
    pair0 = data['train'][0]
    input_grid = pair0['input']
    output_grid = pair0['output']
    
    # Create the visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Input grid
    axes[0, 0].imshow(input_grid, cmap='tab10', vmin=0, vmax=9)
    axes[0, 0].set_title('Input 8x8')
    for r in range(8):
        for c in range(8):
            axes[0, 0].text(c, r, str(input_grid[r][c]), ha='center', va='center', fontsize=8)
    
    # Output grid
    axes[0, 1].imshow(output_grid, cmap='tab10', vmin=0, vmax=9)  
    axes[0, 1].set_title('Output 16x16')
    for r in range(0, 16, 2):  # Show every other cell to avoid clutter
        for c in range(0, 16, 2):
            axes[0, 1].text(c, r, str(output_grid[r][c]), ha='center', va='center', fontsize=6)
    
    # Extract 6x6 non-3 region
    non3_region = []
    for r in range(2, 8):
        row = []
        for c in range(2, 8):
            row.append(input_grid[r][c])
        non3_region.append(row)
    
    axes[0, 2].imshow(non3_region, cmap='tab10', vmin=0, vmax=9)
    axes[0, 2].set_title('Extracted 6x6 Non-3 Region')
    for r in range(6):
        for c in range(6):
            axes[0, 2].text(c, r, str(non3_region[r][c]), ha='center', va='center', fontsize=10)
    
    # Show the 4 quadrants of output
    quadrants = [
        ("Top-Left", [output_grid[r][0:6] for r in range(0, 6)]),
        ("Top-Right", [output_grid[r][10:16] for r in range(0, 6)]),
        ("Bottom-Left", [output_grid[r][0:6] for r in range(10, 16)])
    ]
    
    for i, (title, quad) in enumerate(quadrants):
        axes[1, i].imshow(quad, cmap='tab10', vmin=0, vmax=9)
        axes[1, i].set_title(title + ' 6x6')
        for r in range(6):
            for c in range(6):
                axes[1, i].text(c, r, str(quad[r][c]), ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/Users/evanpieser/76d965ef_viz.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved to 76d965ef_viz.png")
    
    # Analyze the pattern
    print("\n=== PATTERN ANALYSIS ===")
    print("Original 6x6 region:")
    for i, row in enumerate(non3_region):
        print(f"  {i}: {row}")
    
    # Check specific patterns
    print("\n1. Bottom-Right quadrant (rows 10-15, cols 10-15): EXACT COPY")
    
    print("\n2. Top-Left quadrant analysis:")
    tl_quad = [output_grid[r][0:6] for r in range(0, 6)]
    print("   Top-Left:")
    for i, row in enumerate(tl_quad):
        print(f"    {i}: {row}")
    
    print("\n3. Top-Right quadrant analysis:")  
    tr_quad = [output_grid[r][10:16] for r in range(0, 6)]
    print("   Top-Right:")
    for i, row in enumerate(tr_quad):
        print(f"    {i}: {row}")
        
    print("\n4. Bottom-Left quadrant analysis:")
    bl_quad = [output_grid[r][0:6] for r in range(10, 16)]
    print("   Bottom-Left:")
    for i, row in enumerate(bl_quad):
        print(f"    {i}: {row}")
        
    # Check if TL is transpose of original
    print("\n=== TRANSFORMATION ANALYSIS ===")
    print("Checking if Top-Left is transpose:")
    for i in range(6):
        for j in range(6):
            if tl_quad[i][j] != non3_region[j][i]:
                print(f"  Not transpose at ({i},{j}): {tl_quad[i][j]} != {non3_region[j][i]}")
                break
    else:
        print("  Top-Left IS transpose of original!")
        
    # Check if TR is different transformation
    print("\nChecking Top-Right pattern...")
    # Let me see if it's related to the last row/col of original
    print("  Original bottom row:", non3_region[5])
    print("  Original right col:", [non3_region[i][5] for i in range(6)])
    
    return non3_region, tl_quad, tr_quad, bl_quad

if __name__ == "__main__":
    original, tl, tr, bl = visualize_task()