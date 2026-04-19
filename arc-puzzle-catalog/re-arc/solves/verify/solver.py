"""Verify and run an ARC solver against task data."""
import json, sys, importlib.util, os

def load_solver(solver_path):
    spec = importlib.util.spec_from_file_location("solver", solver_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.transform

def verify_and_solve(task_path, solver_path):
    with open(task_path) as f:
        task = json.load(f)
    
    transform = load_solver(solver_path)
    
    # Verify on training data
    for i, ex in enumerate(task['train']):
        pred = transform(ex['input'])
        if pred != ex['output']:
            print(f"FAIL train[{i}]: shapes pred={len(pred)}x{len(pred[0]) if pred else 0} vs expected={len(ex['output'])}x{len(ex['output'][0])}")
            # Show first difference
            for r in range(min(len(pred), len(ex['output']))):
                for c in range(min(len(pred[r]), len(ex['output'][r]))):
                    if pred[r][c] != ex['output'][r][c]:
                        print(f"  First diff at ({r},{c}): got {pred[r][c]} expected {ex['output'][r][c]}")
                        break
                else:
                    continue
                break
            return False
        print(f"PASS train[{i}]")
    
    # Apply to test inputs
    results = []
    for i, t in enumerate(task['test']):
        pred = transform(t['input'])
        print(f"test[{i}]: {len(pred)}x{len(pred[0]) if pred else 0}")
        results.append(pred)
    
    # Save results
    tid = os.path.basename(task_path).replace('.json', '')
    out_path = f'/tmp/rearc_agent_solves/{tid}_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"VERIFIED — saved to {out_path}")
    return True

if __name__ == '__main__':
    verify_and_solve(sys.argv[1], sys.argv[2])
