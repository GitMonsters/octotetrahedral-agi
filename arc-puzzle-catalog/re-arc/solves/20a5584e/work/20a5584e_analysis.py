import json
import numpy as np

def analyze_differences():
    """Analyze what changes between input and output for each training pair"""
    
    with open('/Users/evanpieser/apr12_tasks/20a5584e.json', 'r') as f:
        task = json.load(f)
    
    for i, pair in enumerate(task['train']):
        print(f"\n=== TRAINING PAIR {i+1} ===")
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        # Find dimensions
        height, width = input_grid.shape
        print(f"Grid size: {height}x{width}")
        
        # Find all colors present
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())
        print(f"Input colors: {sorted(input_colors)}")
        print(f"Output colors: {sorted(output_colors)}")
        
        # Find background color (most frequent)
        from collections import Counter
        color_counts = Counter(input_grid.flatten())
        background = max(color_counts, key=color_counts.get)
        print(f"Background color: {background}")
        
        # Find positions of color 1 (blue dots)
        ones_positions = []
        for r in range(height):
            for c in range(width):
                if input_grid[r, c] == 1:
                    ones_positions.append((r, c))
        print(f"Positions of 1s (blue dots): {ones_positions}")
        
        # Find differences between input and output
        differences = []
        for r in range(height):
            for c in range(width):
                if input_grid[r, c] != output_grid[r, c]:
                    differences.append((r, c, input_grid[r, c], output_grid[r, c]))
        
        print(f"Total changes: {len(differences)}")
        print("Changes (row, col, from_color, to_color):")
        for diff in differences[:10]:  # Show first 10 changes
            print(f"  {diff}")
        if len(differences) > 10:
            print(f"  ... and {len(differences) - 10} more")
        
        # Find existing patterns/shapes (non-background, non-1 colors)
        pattern_colors = input_colors - {background, 1}
        print(f"Pattern colors (non-bg, non-1): {sorted(pattern_colors)}")
        
        # Find positions of pattern colors
        for pattern_color in pattern_colors:
            positions = []
            for r in range(height):
                for c in range(width):
                    if input_grid[r, c] == pattern_color:
                        positions.append((r, c))
            print(f"Positions of color {pattern_color}: {positions}")

if __name__ == "__main__":
    analyze_differences()