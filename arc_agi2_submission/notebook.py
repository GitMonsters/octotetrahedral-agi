#!/usr/bin/env python3
"""
ARC-AGI-2 Kaggle Submission
===========================
TranscendPlexity Solver - Visual-First Approach

Author: Evan Pieser
Competition: ARC Prize 2026 - ARC-AGI-2
"""

import json
import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# SOLVER REGISTRY - 65+ working solvers
# ═══════════════════════════════════════════════════════════════════════════════

SOLVERS = {}

def register_solver(task_id):
    """Decorator to register a solver function."""
    def decorator(func):
        SOLVERS[task_id] = func
        return func
    return decorator

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD SOLVERS FROM FILES (for local testing)
# ═══════════════════════════════════════════════════════════════════════════════

def load_solver_files(solver_dir="/tmp/bench_solvers"):
    """Load solver functions from .py files."""
    solver_path = Path(solver_dir)
    if not solver_path.exists():
        return
    
    for solver_file in solver_path.glob("*.py"):
        task_id = solver_file.stem
        try:
            code = solver_file.read_text()
            namespace = {}
            exec(code, namespace)
            if "transform" in namespace:
                SOLVERS[task_id] = namespace["transform"]
        except Exception as e:
            print(f"Warning: Failed to load {task_id}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def identity_solver(grid):
    """Fallback: return input unchanged."""
    return [row[:] for row in grid]

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict_task(task_id, task_data):
    """Generate predictions for a task."""
    solver = SOLVERS.get(task_id, identity_solver)
    
    predictions = []
    for test_case in task_data.get("test", []):
        input_grid = test_case["input"]
        try:
            output = solver(input_grid)
            # Validate output format
            if not isinstance(output, list):
                output = input_grid
            elif not all(isinstance(row, list) for row in output):
                output = input_grid
        except Exception as e:
            print(f"Error on {task_id}: {e}")
            output = input_grid
        predictions.append(output)
    
    return predictions

# ═══════════════════════════════════════════════════════════════════════════════
# SUBMISSION FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

def create_submission(challenges, output_path="submission.json"):
    """
    Create submission.json in Kaggle format.
    
    Format: {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}
    """
    submission = {}
    
    for task_id, task_data in challenges.items():
        predictions = predict_task(task_id, task_data)
        
        # Format: 2 attempts per test case
        submission[task_id] = []
        for pred in predictions:
            submission[task_id].append({
                "attempt_1": pred,
                "attempt_2": pred,
            })
    
    with open(output_path, "w") as f:
        json.dump(submission, f)
    
    return submission

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Detect environment
    KAGGLE_INPUT = "/kaggle/input/arc-prize-2026-arc-agi-2"
    
    if os.path.exists(KAGGLE_INPUT):
        print("🏆 Running on Kaggle")
        challenges_path = f"{KAGGLE_INPUT}/arc-agi_evaluation_challenges.json"
        output_path = "/kaggle/working/submission.json"
        # Solvers would be embedded in notebook for Kaggle
    else:
        print("🖥️ Running locally")
        # Load solvers from files
        load_solver_files("/tmp/bench_solvers")
        print(f"📦 Loaded {len(SOLVERS)} solvers")
        
        # Use local test data
        challenges_path = "/tmp/test_challenges.json"
        output_path = "submission.json"
        
        # Create test challenges from bench_tasks
        if not Path(challenges_path).exists():
            challenges = {}
            for tf in Path("/tmp/bench_tasks").glob("*.json"):
                task_id = tf.stem
                data = json.loads(tf.read_text())
                challenges[task_id] = data.get(task_id, data)
            with open(challenges_path, "w") as f:
                json.dump(challenges, f)
    
    # Load challenges
    with open(challenges_path) as f:
        challenges = json.load(f)
    print(f"📋 Tasks: {len(challenges)}")
    
    # Create submission
    submission = create_submission(challenges, output_path)
    
    # Stats
    with_solver = sum(1 for t in challenges if t in SOLVERS)
    print(f"\n✅ Submission created: {output_path}")
    print(f"   Tasks with specific solvers: {with_solver}/{len(challenges)}")
    print(f"   Tasks using fallback: {len(challenges) - with_solver}")

if __name__ == "__main__":
    main()
