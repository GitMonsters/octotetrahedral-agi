"""
Test suite for ARC puzzle c8b7cc0f solver.
Run with: python3 test_solver.py
"""

import json
from solver import solve

def test_all_training_examples():
    """Test solver on all training examples."""
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/c8b7cc0f.json') as f:
        task = json.load(f)
    
    print("=" * 80)
    print("ARC Puzzle c8b7cc0f - Test Results")
    print("=" * 80)
    
    all_passed = True
    
    for idx, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = solve(inp)
        
        passed = result == expected
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\nTraining Example {idx}: {status}")
        
        if not passed:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("Result: ✓ ALL TESTS PASSED")
        return True
    else:
        print("Result: ✗ SOME TESTS FAILED")
        return False

if __name__ == '__main__':
    success = test_all_training_examples()
    exit(0 if success else 1)
