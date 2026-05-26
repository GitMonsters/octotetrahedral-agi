#!/usr/bin/env python3
import json
import sys

# Load the solver function
exec(open('/Users/evanpieser/4e600a86_solver_v1.py').read())

# Load task data
with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
    task = json.load(f)

def test_solver():
    """Test the solve function on all training examples"""
    train_pairs = task['train']
    
    total_correct = 0
    total_examples = len(train_pairs)
    
    for i, pair in enumerate(train_pairs):
        print(f"\n{'='*50}")
        print(f"TESTING TRAIN PAIR {i+1}")
        print(f"{'='*50}")
        
        input_grid = pair['input']
        expected_output = pair['output']
        
        # Test our solver
        predicted_output = solve(input_grid)
        
        # Compare with expected
        if predicted_output == expected_output:
            print("✅ CORRECT!")
            total_correct += 1
        else:
            print("❌ INCORRECT")
            
            # Show differences
            h, w = len(input_grid), len(input_grid[0])
            differences = []
            
            for r in range(h):
                for c in range(w):
                    if predicted_output[r][c] != expected_output[r][c]:
                        differences.append((r, c, predicted_output[r][c], expected_output[r][c]))
            
            print(f"Found {len(differences)} differences:")
            for r, c, pred, exp in differences[:10]:  # Show first 10
                print(f"  ({r},{c}): predicted={pred}, expected={exp}")
            
            if len(differences) > 10:
                print(f"  ... and {len(differences) - 10} more")
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS: {total_correct}/{total_examples} correct")
    print(f"Success rate: {total_correct/total_examples*100:.1f}%")
    print(f"{'='*50}")
    
    return total_correct == total_examples

if __name__ == "__main__":
    success = test_solver()
    if not success:
        print("\n❌ Solver needs debugging!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")