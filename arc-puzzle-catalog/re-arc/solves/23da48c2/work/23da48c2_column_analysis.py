#!/usr/bin/env python3

import json
import numpy as np

def analyze_column_removal():
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        task = json.load(f)
    
    print("=== COLUMN REMOVAL ANALYSIS ===")
    print("Since this is a column-reduction task, let's see which columns are kept/removed")
    
    for i, pair in enumerate(task['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"\n--- TRAIN {i} ---")
        print(f"Input: {input_grid.shape}, Output: {output_grid.shape}")
        print(f"Columns: {input_grid.shape[1]} -> {output_grid.shape[1]} (removed {input_grid.shape[1] - output_grid.shape[1]})")
        
        # Since rows are the same, let's find which input columns map to output columns
        # We'll try to match each output column to an input column
        
        mapping = []
        for out_col in range(output_grid.shape[1]):
            output_column = output_grid[:, out_col]
            
            best_match_col = None
            best_match_score = 0
            
            # Compare with all input columns
            for in_col in range(input_grid.shape[1]):
                input_column = input_grid[:, in_col]
                
                # Count exact matches
                matches = np.sum(input_column == output_column)
                score = matches / len(output_column)
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_col = in_col
            
            mapping.append((out_col, best_match_col, best_match_score))
            if best_match_score == 1.0:
                print(f"  Output col {out_col:2d} = Input col {best_match_col:2d} (exact match)")
            else:
                print(f"  Output col {out_col:2d} ≈ Input col {best_match_col:2d} (match: {best_match_score:.2f})")
        
        # Analyze which input columns were kept
        kept_columns = [m[1] for m in mapping if m[2] >= 0.8]  # High confidence matches
        removed_columns = [c for c in range(input_grid.shape[1]) if c not in kept_columns]
        
        print(f"  Kept input columns: {kept_columns}")
        print(f"  Removed columns: {removed_columns}")
        print(f"  Kept: {len(kept_columns)}, Removed: {len(removed_columns)}")
        
        # Look for patterns in removed columns
        if len(removed_columns) > 0:
            print(f"  Removed column analysis:")
            print(f"    First removed: {min(removed_columns)}, Last removed: {max(removed_columns)}")
            
            # Check if removed columns are contiguous
            removed_set = set(removed_columns)
            contiguous_ranges = []
            start = None
            for col in range(input_grid.shape[1]):
                if col in removed_set:
                    if start is None:
                        start = col
                else:
                    if start is not None:
                        contiguous_ranges.append((start, col-1))
                        start = None
            if start is not None:
                contiguous_ranges.append((start, input_grid.shape[1]-1))
            
            print(f"    Contiguous removed ranges: {contiguous_ranges}")

if __name__ == "__main__":
    analyze_column_removal()