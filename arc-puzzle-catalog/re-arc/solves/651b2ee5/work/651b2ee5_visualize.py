#!/usr/bin/env python3

import json
import numpy as np
from PIL import Image, ImageDraw
import colorsys

def get_color(value):
    """Convert ARC color number to RGB"""
    colors = {
        0: (0, 0, 0),      # black
        1: (0, 116, 217),  # blue
        2: (255, 65, 54),  # red
        3: (46, 204, 64),  # green
        4: (255, 220, 0),  # yellow
        5: (170, 170, 170), # gray
        6: (240, 18, 190), # magenta
        7: (255, 133, 27), # orange
        8: (127, 219, 255), # sky blue
        9: (135, 12, 37)   # maroon
    }
    return colors.get(value, (128, 128, 128))

def visualize_grid(grid, title="Grid"):
    """Create PIL image from grid"""
    grid = np.array(grid)
    height, width = grid.shape
    
    # Scale up for visibility
    scale = 20
    img = Image.new('RGB', (width * scale, height * scale), 'white')
    
    for y in range(height):
        for x in range(width):
            color = get_color(grid[y, x])
            for dy in range(scale):
                for dx in range(scale):
                    img.putpixel((x * scale + dx, y * scale + dy), color)
    
    return img

def visualize_task():
    # Load task data
    with open('/Users/evanpieser/apr12_tasks/651b2ee5.json', 'r') as f:
        task = json.load(f)
    
    train_pairs = task['train']
    
    # Calculate total image dimensions
    max_width = 0
    total_height = 0
    
    pair_images = []
    for i, pair in enumerate(train_pairs):
        input_grid = pair['input']
        output_grid = pair['output']
        
        input_img = visualize_grid(input_grid, f"Train {i+1} Input")
        output_img = visualize_grid(output_grid, f"Train {i+1} Output")
        
        # Combine input and output horizontally
        combined_width = input_img.width + output_img.width + 40  # 40px gap
        combined_height = max(input_img.height, output_img.height)
        
        combined = Image.new('RGB', (combined_width, combined_height), 'white')
        combined.paste(input_img, (0, 0))
        combined.paste(output_img, (input_img.width + 40, 0))
        
        pair_images.append(combined)
        max_width = max(max_width, combined_width)
        total_height += combined_height + 20  # 20px gap between pairs
    
    # Create final visualization
    final_img = Image.new('RGB', (max_width, total_height), 'white')
    y_offset = 0
    
    for img in pair_images:
        final_img.paste(img, (0, y_offset))
        y_offset += img.height + 20
    
    final_img.save('651b2ee5_viz.png')
    print("Visualization saved to 651b2ee5_viz.png")
    
    # Also print the data for analysis
    print("\n=== TRAINING PAIR ANALYSIS ===")
    for i, pair in enumerate(train_pairs):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"\nTrain pair {i+1}:")
        print(f"Input shape: {input_grid.shape}")
        print(f"Output shape: {output_grid.shape}")
        
        # Find marker positions (non-background pixels)
        unique_input = np.unique(input_grid)
        print(f"Input colors: {unique_input}")
        
        markers = []
        for color in unique_input:
            if color != 2:  # 2 seems to be background in some cases
                positions = np.where(input_grid == color)
                for y, x in zip(positions[0], positions[1]):
                    markers.append((y, x, color))
        
        print(f"Marker positions: {markers}")
        
        # Print input grid
        print("Input:")
        for row in input_grid:
            print(' '.join(f'{x:2d}' for x in row))
        
        print("Output:")
        for row in output_grid:
            print(' '.join(f'{x:2d}' for x in row))

if __name__ == "__main__":
    visualize_task()