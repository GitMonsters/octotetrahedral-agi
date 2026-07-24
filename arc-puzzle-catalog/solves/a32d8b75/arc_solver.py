from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from arc_utils import TASK_ID, TASK_PATH, count_differences, ensure_rectangular, load_task, save_json
from solver import solve


def validate_training_examples(task: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, example in enumerate(task['train']):
        prediction = solve(example['input'])
        ensure_rectangular(prediction)
        diff_count = count_differences(prediction, example['output'])
        results.append(
            {
                'split': 'train',
                'index': index,
                'passed': prediction == example['output'],
                'differences': diff_count,
                'prediction': prediction,
            }
        )
    return results


def generate_test_predictions(task: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, example in enumerate(task['test']):
        prediction = solve(example['input'])
        ensure_rectangular(prediction)
        payload: dict[str, Any] = {
            'split': 'test',
            'index': index,
            'prediction': prediction,
        }
        if 'output' in example:
            payload['passed_reference'] = prediction == example['output']
            payload['differences'] = count_differences(prediction, example['output'])
        results.append(payload)
    return results


def build_results(task: dict[str, Any]) -> dict[str, Any]:
    train_results = validate_training_examples(task)
    test_results = generate_test_predictions(task)
    accuracy = 0.0
    if train_results:
        accuracy = sum(1 for result in train_results if result['passed']) / len(train_results)
    return {
        'task_id': TASK_ID,
        'task_path': str(TASK_PATH),
        'training_accuracy': accuracy,
        'train_results': train_results,
        'test_results': test_results,
    }


def run(task_path: str | Path = TASK_PATH, results_path: str | Path | None = None) -> dict[str, Any]:
    task = load_task(task_path)
    results = build_results(task)
    if results_path is not None:
        save_json(results, results_path)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f'Solve ARC task {TASK_ID}')
    parser.add_argument('--task', default=str(TASK_PATH), help='Path to the task JSON file')
    parser.add_argument('--results', help='Optional path for saving JSON results')
    args = parser.parse_args(argv)

    results = run(args.task, args.results)
    for result in results['train_results']:
        status = 'PASS' if result['passed'] else 'FAIL'
        print(f"Train {result['index']}: {status} ({result['differences']} diffs)")
    print(f"Training accuracy: {results['training_accuracy']:.0%}")
    for result in results['test_results']:
        extra = ''
        if 'passed_reference' in result:
            extra = f" reference={'PASS' if result['passed_reference'] else 'FAIL'}"
        print(f"Test {result['index']}: {len(result['prediction'])}x{len(result['prediction'][0])}{extra}")
    if args.results:
        print(f"Saved results to {args.results}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
