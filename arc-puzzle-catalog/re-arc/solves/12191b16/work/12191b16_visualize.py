#!/usr/bin/env python3

import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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
    9: (128, 0, 0),      # maroon
}

def load_task():
    with open('/Users/evanpieser/apr12_tasks/12191b16.json', 'r') as f:
        return json.load(f)

def grid_to_image(grid, cell_size=20):
    """Convert a 2D grid to a PIL image"""
    rows, cols = len(grid), len(grid[0])
    img = Image.new('RGB', (cols * cell_size, rows * cell_size), 'white')
    
    for r in range(rows):
        for c in range(cols):
            color = color_map.get(grid[r][c], (255, 255, 255))  # default to white
            x1, y1 = c * cell_size, r * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            
            # Fill the cell
            for x in range(x1, x2):
                for y in range(y1, y2):
                    img.putpixel((x, y), color)
    
    return img

def create_visualization():
    task = load_task()
    cell_size = 15
    margin = 20
    
    # Calculate layout
    train_pairs = task['train']
    num_pairs = len(train_pairs)
    
    # Find max dimensions
    max_width = 0
    max_height = 0
    
    for pair in train_pairs:
        input_grid = pair['input']
        output_grid = pair['output']
        
        max_width = max(max_width, len(input_grid[0]) * 2 + 2)  # input + output + gap
        max_height = max(max_height, len(input_grid))
    
    # Create canvas
    canvas_width = max_width * cell_size + margin * 2
    canvas_height = (max_height * num_pairs + margin * (num_pairs + 1)) * cell_size + margin * 2
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    
    # Draw each training pair
    y_offset = margin
    
    for i, pair in enumerate(train_pairs):
        input_grid = pair['input']
        output_grid = pair['output']
        
        # Draw input grid
        input_img = grid_to_image(input_grid, cell_size)
        canvas.paste(input_img, (margin, y_offset))
        
        # Draw output grid
        output_img = grid_to_image(output_grid, cell_size)
        input_width = len(input_grid[0]) * cell_size
        canvas.paste(output_img, (margin + input_width + margin, y_offset))
        
        # Add labels
        try:
            draw = ImageDraw.Draw(canvas)
            draw.text((margin, y_offset - 20), f"Train {i+1} - Input", fill='black')
            draw.text((margin + input_width + margin, y_offset - 20), f"Output", fill='black')
        except:
            pass  # Skip text if no font available
        
        y_offset += len(input_grid) * cell_size + margin
    
    # Save the visualization
    canvas.save('12191b16_viz.png')
    print("Visualization saved as 12191b16_viz.png")

if __name__ == "__main__":
    create_visualization()