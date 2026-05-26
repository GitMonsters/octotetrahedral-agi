#!/usr/bin/env python3

import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load the task
with open('/Users/evanpieser/apr12_tasks/20a5584e.json', 'r') as f:
    task = json.load(f)

# Color mapping: 0=black,1=blue,2=red,3=green,4=yellow,5=gray,6=magenta,7=orange,8=cyan,9=maroon
color_map = {
    0: (0, 0, 0),        # black
    1: (0, 0, 255),      # blue
    2: (255, 0, 0),      # red
    3: (0, 255, 0),      # green
    4: (255, 255, 0),    # yellow
    5: (128, 128, 128),  # gray
    6: (255, 0, 255),    # magenta
    7: (255, 165, 0),    # orange
    8: (0, 255, 255),    # cyan
    9: (128, 0, 0)       # maroon
}

def grid_to_image(grid, cell_size=20):
    """Convert grid to PIL Image"""
    height = len(grid)
    width = len(grid[0])
    
    img = Image.new('RGB', (width * cell_size, height * cell_size), 'white')
    
    for r in range(height):
        for c in range(width):
            color = color_map.get(grid[r][c], (128, 128, 128))  # Default to gray
            x1 = c * cell_size
            y1 = r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            # Fill the cell
            for x in range(x1, x2):
                for y in range(y1, y2):
                    img.putpixel((x, y), color)
    
    return img

def create_visualization():
    """Create a complete visualization of all train pairs"""
    
    train_pairs = task['train']
    num_pairs = len(train_pairs)
    
    # Find max dimensions for consistent sizing
    max_height = 0
    max_width = 0
    
    for pair in train_pairs:
        for grid in [pair['input'], pair['output']]:
            max_height = max(max_height, len(grid))
            max_width = max(max_width, len(grid[0]))
    
    cell_size = 15
    grid_spacing = 30
    pair_spacing = 50
    
    # Calculate total image dimensions
    # Each pair: input + arrow + output
    single_grid_w = max_width * cell_size
    single_grid_h = max_height * cell_size
    arrow_width = 50
    
    pair_width = single_grid_w + arrow_width + single_grid_w
    total_width = max(800, pair_width + 100)
    
    # Arrange pairs vertically
    total_height = num_pairs * (single_grid_h + pair_spacing) + 100
    
    # Create main image
    img = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(img)
    
    y_offset = 50
    
    for i, pair in enumerate(train_pairs):
        # Draw pair label
        draw.text((50, y_offset - 20), f"Training Pair {i+1}", fill=(0, 0, 0))
        
        input_grid = pair['input']
        output_grid = pair['output']
        
        # Convert grids to images
        input_img = grid_to_image(input_grid, cell_size)
        output_img = grid_to_image(output_grid, cell_size)
        
        # Position input
        x_input = 50
        img.paste(input_img, (x_input, y_offset))
        
        # Draw arrow
        arrow_x = x_input + single_grid_w + 10
        arrow_y = y_offset + single_grid_h // 2
        draw.text((arrow_x, arrow_y - 10), "→", fill=(0, 0, 0))
        
        # Position output
        x_output = arrow_x + arrow_width
        img.paste(output_img, (x_output, y_offset))
        
        # Add input/output labels
        draw.text((x_input, y_offset + single_grid_h + 5), "Input", fill=(0, 0, 0))
        draw.text((x_output, y_offset + single_grid_h + 5), "Output", fill=(0, 0, 0))
        
        y_offset += single_grid_h + pair_spacing
    
    return img

if __name__ == "__main__":
    viz_img = create_visualization()
    viz_img.save('/Users/evanpieser/20a5584e_viz.png')
    print("Visualization saved to 20a5584e_viz.png")