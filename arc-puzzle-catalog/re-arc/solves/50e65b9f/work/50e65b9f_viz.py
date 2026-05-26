import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def load_task(task_file):
    with open(task_file, 'r') as f:
        return json.load(f)

def grid_to_image(grid, cell_size=20):
    """Convert a grid to a PIL Image with color mapping"""
    # ARC color palette
    colors = {
        0: (0, 0, 0),        # black
        1: (0, 116, 217),    # blue  
        2: (255, 65, 54),    # red
        3: (46, 204, 64),    # green
        4: (255, 220, 0),    # yellow
        5: (170, 170, 170),  # gray
        6: (240, 18, 190),   # magenta
        7: (255, 133, 27),   # orange
        8: (127, 219, 255),  # cyan
        9: (135, 12, 37),    # brown
    }
    
    height, width = len(grid), len(grid[0])
    img = Image.new('RGB', (width * cell_size, height * cell_size), 'white')
    draw = ImageDraw.Draw(img)
    
    for r in range(height):
        for c in range(width):
            color = colors.get(grid[r][c], (128, 128, 128))
            x1, y1 = c * cell_size, r * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            draw.rectangle([x1, y1, x2, y2], fill=color, outline='black')
    
    return img

def create_visualization(task_data):
    """Create a visualization showing all train pairs"""
    train_pairs = task_data['train']
    
    # Calculate image dimensions
    max_width = 0
    total_height = 0
    
    for pair in train_pairs:
        input_grid = pair['input']
        output_grid = pair['output']
        pair_width = len(input_grid[0]) + len(output_grid[0]) + 2  # +2 for separator
        max_width = max(max_width, pair_width)
        total_height += max(len(input_grid), len(output_grid)) + 2  # +2 for spacing
    
    cell_size = 20
    img_width = max_width * cell_size + 100  # Extra space for labels
    img_height = total_height * cell_size + 100
    
    # Create the main image
    main_img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(main_img)
    
    y_offset = 20
    
    for i, pair in enumerate(train_pairs):
        input_grid = pair['input']
        output_grid = pair['output']
        
        # Draw labels
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, y_offset), f"Train {i+1}:", fill='black', font=font)
        draw.text((10, y_offset + 20), "Input", fill='black', font=font)
        
        # Create input image
        input_img = grid_to_image(input_grid, cell_size)
        main_img.paste(input_img, (10, y_offset + 40))
        
        # Create output image
        output_img = grid_to_image(output_grid, cell_size)
        arrow_x = 20 + len(input_grid[0]) * cell_size
        draw.text((arrow_x, y_offset + 20), "→ Output", fill='black', font=font)
        main_img.paste(output_img, (arrow_x + 50, y_offset + 40))
        
        y_offset += max(len(input_grid), len(output_grid)) * cell_size + 80
    
    return main_img

# Load task and create visualization
task_data = load_task('/Users/evanpieser/apr12_tasks/50e65b9f.json')
viz_img = create_visualization(task_data)
viz_img.save('/Users/evanpieser/50e65b9f_viz.png')
print("Visualization saved to 50e65b9f_viz.png")

# Let's also print some analysis
print("\nAnalysis of train pairs:")
for i, pair in enumerate(task_data['train']):
    input_grid = pair['input']
    output_grid = pair['output']
    
    print(f"\nTrain {i+1}:")
    print(f"  Input size: {len(input_grid)}x{len(input_grid[0])}")
    print(f"  Output size: {len(output_grid)}x{len(output_grid[0])}")
    
    # Count colors in input
    input_colors = {}
    for row in input_grid:
        for cell in row:
            input_colors[cell] = input_colors.get(cell, 0) + 1
    
    # Count colors in output
    output_colors = {}
    for row in output_grid:
        for cell in row:
            output_colors[cell] = output_colors.get(cell, 0) + 1
    
    print(f"  Input colors: {dict(sorted(input_colors.items()))}")
    print(f"  Output colors: {dict(sorted(output_colors.items()))}")
    
    # Find changed cells
    changes = []
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r][c] != output_grid[r][c]:
                changes.append((r, c, input_grid[r][c], output_grid[r][c]))
    
    print(f"  Changed cells: {len(changes)}")
    if len(changes) <= 20:  # Only show if not too many
        for r, c, old, new in changes[:10]:
            print(f"    ({r},{c}): {old} → {new}")
        if len(changes) > 10:
            print(f"    ... and {len(changes)-10} more")