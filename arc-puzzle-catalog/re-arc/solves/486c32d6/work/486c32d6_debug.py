#!/usr/bin/env python3
import json

# Load task
with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
    task = json.load(f)

def debug_train_0():
    print("=== DEBUG TRAIN 0 ===")
    example = task['train'][0]
    input_grid = example['input']
    output_grid = example['output']
    
    # Focus on row 1 that changed
    r = 1
    input_row = input_grid[r]
    output_row = output_grid[r]
    
    print(f"Input row {r}:  {input_row}")
    print(f"Output row {r}: {output_row}")
    
    # Find separator columns
    separators = [3, 7, 11, 15, 19]  # Known from previous analysis
    print(f"Separators: {separators}")
    
    # Define cells
    cell_boundaries = [0, 4, 8, 12, 16, 20, 23]
    cells = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 23)]
    
    print("Cell analysis:")
    for i, (start, end) in enumerate(cells):
        cell_input = []
        cell_output = []
        for c in range(start, end):
            if c not in separators:
                cell_input.append(input_row[c])
                cell_output.append(output_row[c])
        
        print(f"  Cell {i} [{start}:{end}]: input={cell_input}, output={cell_output}")
        
        # Check what changed
        if cell_input != cell_output:
            for j, (inp, out) in enumerate(zip(cell_input, cell_output)):
                if inp != out:
                    print(f"    Position {j}: {inp} -> {out}")

def debug_train_1():
    print("\n=== DEBUG TRAIN 1 ===")
    example = task['train'][1]
    input_grid = example['input']
    output_grid = example['output']
    
    # Look at row 1 (has 9 anomaly)
    r = 1
    input_row = input_grid[r]
    output_row = output_grid[r]
    
    print(f"Input row {r}:  {input_row}")
    print(f"Output row {r}: {output_row}")
    
    # Look at other rows with 9
    print("\nLooking for pattern with value 9:")
    for r in range(len(input_grid)):
        if 9 in input_grid[r]:
            print(f"Row {r} input:  {input_grid[r]}")
            print(f"Row {r} output: {output_grid[r]}")

def debug_train_2():
    print("\n=== DEBUG TRAIN 2 ===")
    example = task['train'][2]
    input_grid = example['input']
    output_grid = example['output']
    
    # Look at row 6 (complex changes)
    r = 6
    input_row = input_grid[r]
    output_row = output_grid[r]
    
    print(f"Input row {r}:  {input_row}")
    print(f"Output row {r}: {output_row}")
    
    # Find separator columns
    separators = [2, 5, 8, 11, 14, 17]  # Known from previous analysis
    print(f"Separators: {separators}")
    
    # Define cells
    cells = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 20)]
    
    print("Cell analysis:")
    for i, (start, end) in enumerate(cells):
        cell_input = []
        cell_output = []
        for c in range(start, end):
            if c not in separators:
                cell_input.append(input_row[c])
                cell_output.append(output_row[c])
        
        print(f"  Cell {i} [{start}:{end}]: input={cell_input}, output={cell_output}")

debug_train_0()
debug_train_1()
debug_train_2()