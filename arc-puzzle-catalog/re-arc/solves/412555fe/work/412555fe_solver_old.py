#!/usr/bin/env python3
"""
ARC-AGI Task 412555fe Solver
"""

def transform(grid):
    """
    Transform the grid by filling cells to achieve 180-degree rotational symmetry.
    Priority given to non-filler values.
    """
    import numpy as np
    
    arr = np.array(grid)
    h, w = arr.shape
    
    # Find the filler color (most common)
    unique, counts = np.unique(arr, return_counts=True)
    filler = unique[np.argmax(counts)]
    
    result = arr.copy()
    
    # Process each cell and its 180-degree symmetric partner
    for y in range((h + 1) // 2):
        for x in range(w):
            y_sym = h - 1 - y
            x_sym = w - 1 - x
            
            val1 = arr[y, x]
            val2 = arr[y_sym, x_sym]
            
            # If values differ and one is filler, make them match with non-filler value
            if val1 != val2:
                if val1 == filler and val2 != filler:
                    result[y, x] = val2
                elif val2 == filler and val1 != filler:
                    result[y_sym, x_sym] = val1
    
    return result.tolist()

if __name__ == "__main__":
    import json
    import numpy as np
    
    with open('/tmp/rearc45/412555fe.json', 'r') as f:
        task_data = json.load(f)
    
    print("Testing on training pairs:")
    
    for idx, pair in enumerate(task_data['train']):
        predicted = transform(pair['input'])
        expected = pair['output']
        
        pred_arr = np.array(predicted)
        exp_arr = np.array(expected)
        
        matches = np.sum(pred_arr == exp_arr)
        total = exp_arr.size
        
        status = "PASS" if matches == total else "FAIL"
        
        print(f"  Training pair {idx}: {status} ({matches}/{total})")

