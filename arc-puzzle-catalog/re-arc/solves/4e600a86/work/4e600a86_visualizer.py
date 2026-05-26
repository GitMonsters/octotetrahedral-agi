#!/usr/bin/env python3
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load task data
with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
    task = json.load(f)

# Color mapping for ARC colors
COLORS = {
    0: (0, 0, 0),       # Black
    1: (0, 116, 217),   # Blue
    2: (255, 65, 54),   # Red
    3: (46, 204, 64),   # Green
    4: (255, 220, 0),   # Yellow
    5: (170, 170, 170), # Gray
    6: (240, 18, 190),  # Magenta
    7: (255, 133, 27),  # Orange
    8: (127, 219, 255), # Sky blue
    9: (135, 12, 37)    # Maroon
}

def grid_to_image(grid, cell_size=20):
    """Convert a grid to PIL Image"""
    h, w = len(grid), len(grid[0])
    img = Image.new('RGB', (w * cell_size, h * cell_size), 'white')
    draw = ImageDraw.Draw(img)
    
    for r in range(h):
        for c in range(w):
            color = COLORS.get(grid[r][c], (128, 128, 128))
            x1, y1 = c * cell_size, r * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            draw.rectangle([x1, y1, x2-1, y2-1], fill=color, outline='gray')
    
    return img

def create_visualization():
    """Create visualization of all training examples"""
    train_pairs = task['train']
    
    # Calculate dimensions needed
    cell_size = 15
    max_width = 0
    max_height = 0
    
    for pair in train_pairs:
        input_grid = pair['input']
        output_grid = pair['output']
        h, w = len(input_grid), len(input_grid[0])
        max_width = max(max_width, w * 2 + 2)  # input + output + gap
        max_height = max(max_height, h)
    
    # Create large canvas
    canvas_width = max_width * cell_size
    canvas_height = max_height * len(train_pairs) * cell_size + 100 * len(train_pairs)
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    
    current_y = 10
    
    for i, pair in enumerate(train_pairs):
        input_grid = pair['input']
        output_grid = pair['output']
        
        # Create input and output images
        input_img = grid_to_image(input_grid, cell_size)
        output_img = grid_to_image(output_grid, cell_size)
        
        # Paste on canvas
        canvas.paste(input_img, (10, current_y))
        canvas.paste(output_img, (input_img.width + 30, current_y))
        
        # Add labels
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, current_y - 15), f"Train {i+1} - Input", fill='black', font=font)
        draw.text((input_img.width + 30, current_y - 15), f"Train {i+1} - Output", fill='black', font=font)
        
        current_y += max(input_img.height, output_img.height) + 50
    
    # Save the visualization
    canvas.save('/Users/evanpieser/4e600a86_viz.png')
    print(f"Visualization saved to /Users/evanpieser/4e600a86_viz.png")

if __name__ == "__main__":
    create_visualization()