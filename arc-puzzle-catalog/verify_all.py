#!/usr/bin/env python3
"""Verify all 514 solvers against their test cases."""

import importlib.util
import json
import os
import sys
import traceback

def load_task(task_id: str) -> dict | None:
    path = f"dataset/tasks/{task_id}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_solver(task_id: str):
    path = f"solves/{task_id}/solver.py"
    spec = importlib.util.spec_from_file_location("solver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def verify_task(task_id: str) -> tuple[bool, str]:
    task = load_task(task_id)
    if task is None:
        return False, "task data not found"

    try:
        mod = load_solver(task_id)
    except Exception as e:
        return False, f"load error: {e}"

    if not hasattr(mod, "solve"):
        return False, "no solve() function"

    pairs = task.get("test", [])
    if not pairs:
        return False, "no test pairs"

    for i, pair in enumerate(pairs):
        try:
            result = mod.solve(pair["input"])
        except Exception as e:
            return False, f"test[{i}] runtime error: {e}"

        if result != pair["output"]:
            return False, f"test[{i}] output mismatch"

    return True, f"{len(pairs)} test pair(s) passed"

def main():
    solves_dir = "solves"
    if not os.path.isdir(solves_dir):
        print("Error: solves/ directory not found. Run from repo root.")
        sys.exit(1)

    task_ids = sorted(os.listdir(solves_dir))
    task_ids = [t for t in task_ids if os.path.isdir(os.path.join(solves_dir, t))]

    passed = 0
    failed = 0
    skipped = 0
    failures = []

    print(f"Verifying {len(task_ids)} solvers...\n")

    for task_id in task_ids:
        ok, msg = verify_task(task_id)
        if ok:
            passed += 1
            print(f"  ✅ {task_id}: {msg}")
        else:
            if "not found" in msg:
                skipped += 1
                print(f"  ⏭️  {task_id}: {msg}")
            else:
                failed += 1
                failures.append((task_id, msg))
                print(f"  ❌ {task_id}: {msg}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Total: {len(task_ids)} solvers")

    if failures:
        print(f"\nFailures:")
        for tid, msg in failures:
            print(f"  {tid}: {msg}")
        sys.exit(1)
    else:
        print(f"\n🎉 All {passed} solvers verified!")

if __name__ == "__main__":
    main()
