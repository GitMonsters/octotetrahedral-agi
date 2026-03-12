import json
import os


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Transform rule:
    For each row, extract non-zero values (preserving order)
    and place them right-aligned in the output grid.
    """
    if not grid:
        return grid
    
    width = len(grid[0])
    result = []
    
    for row in grid:
        # Extract all non-zero values in order
        non_zeros = [val for val in row if val != 0]
        
        if not non_zeros:
            # All zeros - keep as is
            result.append([0] * width)
        else:
            # Place non-zeros right-aligned with zeros on the left
            output_row = [0] * (width - len(non_zeros)) + non_zeros
            result.append(output_row)
    
    return result


if __name__ == "__main__":
    # Load the task
    task_path = os.path.expanduser("~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/5ffb2104.json")
    with open(task_path) as f:
        task = json.load(f)
    
    # Test on all training examples
    print("Testing on training examples:")
    all_pass = True
    for idx, example in enumerate(task["train"]):
        inp = example["input"]
        expected = example["output"]
        predicted = solve(inp)
        
        passed = predicted == expected
        all_pass = all_pass and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"  Example {idx + 1}: {status}")
        
        if not passed:
            for i, (exp_row, pred_row) in enumerate(zip(expected, predicted)):
                if exp_row != pred_row:
                    print(f"    Row {i} mismatch:")
                    print(f"      Expected: {exp_row}")
                    print(f"      Got:      {pred_row}")
    
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
