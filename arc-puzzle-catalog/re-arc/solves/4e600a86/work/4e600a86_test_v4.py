#!/usr/bin/env python3
import json
import sys

# Load the solver function
exec(open('/Users/evanpieser/4e600a86_solver_v4.py').read())

# Load task data
with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
    task = json.load(f)

def test_solver_detailed():
    """Test the solve function with detailed analysis"""
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
            
            # Analyze the differences
            h, w = len(input_grid), len(input_grid[0])
            
            # Count actual vs predicted changes
            actual_changes = set()
            predicted_changes = set()
            
            for r in range(h):
                for c in range(w):
                    if input_grid[r][c] != expected_output[r][c]:
                        actual_changes.add((r, c))
                    if input_grid[r][c] != predicted_output[r][c]:
                        predicted_changes.add((r, c))
            
            correct_changes = actual_changes & predicted_changes
            missed_changes = actual_changes - predicted_changes
            extra_changes = predicted_changes - actual_changes
            
            print(f"Expected {len(actual_changes)} changes")
            print(f"Predicted {len(predicted_changes)} changes")
            print(f"Correct: {len(correct_changes)}")
            print(f"Missed: {len(missed_changes)} - {list(missed_changes)[:5]}")
            print(f"Extra: {len(extra_changes)} - {list(extra_changes)[:5]}")
            
            # Calculate accuracy metrics
            if len(predicted_changes) > 0:
                precision = len(correct_changes) / len(predicted_changes)
                recall = len(correct_changes) / len(actual_changes) if len(actual_changes) > 0 else 1.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                print(f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS: {total_correct}/{total_examples} correct")
    print(f"Success rate: {total_correct/total_examples*100:.1f}%")
    print(f"{'='*50}")
    
    return total_correct == total_examples

if __name__ == "__main__":
    success = test_solver_detailed()
    if not success:
        print("\n❌ Solver needs more debugging!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")