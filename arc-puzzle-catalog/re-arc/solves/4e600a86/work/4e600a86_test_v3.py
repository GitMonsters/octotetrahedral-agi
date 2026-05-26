#!/usr/bin/env python3
import json
import sys

# Load the solver function
exec(open('/Users/evanpieser/4e600a86_solver_v3.py').read())

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
            
            # Show detailed differences
            h, w = len(input_grid), len(input_grid[0])
            differences = []
            
            for r in range(h):
                for c in range(w):
                    if predicted_output[r][c] != expected_output[r][c]:
                        differences.append((r, c, predicted_output[r][c], expected_output[r][c]))
            
            print(f"Found {len(differences)} differences:")
            
            # Group differences by type
            false_positives = []  # We predicted 3, but should be background
            false_negatives = []  # We predicted background, but should be 3
            
            for r, c, pred, exp in differences:
                if pred == 3 and exp != 3:
                    false_positives.append((r, c))
                elif pred != 3 and exp == 3:
                    false_negatives.append((r, c))
            
            if false_positives:
                print(f"False positives (predicted 3, should be bg): {len(false_positives)} - {false_positives[:5]}")
            if false_negatives:
                print(f"False negatives (predicted bg, should be 3): {len(false_negatives)} - {false_negatives[:5]}")
            
            # Show accuracy
            total_changes_expected = len([1 for r in range(h) for c in range(w) if input_grid[r][c] != expected_output[r][c]])
            total_changes_predicted = len([1 for r in range(h) for c in range(w) if input_grid[r][c] != predicted_output[r][c]])
            correct_changes = total_changes_expected + total_changes_predicted - len(differences)
            
            print(f"Expected {total_changes_expected} changes, predicted {total_changes_predicted}, got {correct_changes} correct")
    
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