import json
import numpy as np

def analyze_pair(input_grid, output_grid, pair_num):
    print(f"\n=== TRAIN PAIR {pair_num} ANALYSIS ===")
    
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)
    
    # Find background color (most frequent)
    unique, counts = np.unique(input_arr, return_counts=True)
    background_color = unique[np.argmax(counts)]
    print(f"Background color: {background_color}")
    
    # Find all non-background positions and colors
    non_bg_positions = []
    non_bg_colors = []
    
    for i in range(len(input_grid)):
        for j in range(len(input_grid[0])):
            if input_grid[i][j] != background_color:
                non_bg_positions.append((i, j))
                non_bg_colors.append(input_grid[i][j])
                print(f"  Non-background at ({i},{j}): color {input_grid[i][j]}")
    
    print(f"Unique non-background colors: {sorted(set(non_bg_colors))}")
    
    # Find bounding box of non-background elements
    if non_bg_positions:
        min_row = min(pos[0] for pos in non_bg_positions)
        max_row = max(pos[0] for pos in non_bg_positions)
        min_col = min(pos[1] for pos in non_bg_positions)
        max_col = max(pos[1] for pos in non_bg_positions)
        print(f"Bounding box: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
    
    # Analyze output pattern
    print(f"Output grid size: {len(output_grid)}x{len(output_grid[0])}")
    
    # Check if output has regular patterns
    print("Output pattern analysis:")
    for i in range(min(5, len(output_grid))):
        row_colors = [output_grid[i][j] for j in range(min(10, len(output_grid[0])))]
        print(f"  Row {i}: {row_colors}")

def main():
    with open('/Users/evanpieser/apr12_tasks/12191b16.json', 'r') as f:
        task = json.load(f)
    
    for i, pair in enumerate(task['train']):
        analyze_pair(pair['input'], pair['output'], i+1)

if __name__ == "__main__":
    main()