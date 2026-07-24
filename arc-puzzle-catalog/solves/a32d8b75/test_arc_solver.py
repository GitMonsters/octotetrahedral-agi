from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from arc_solver import TASK_PATH, build_results, run
from arc_utils import load_task


class ArcTaskA32D8B75Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = load_task(TASK_PATH)

    def test_training_examples_all_pass(self) -> None:
        results = build_results(self.task)
        self.assertEqual(results['training_accuracy'], 1.0)
        self.assertTrue(all(result['passed'] for result in results['train_results']))

    def test_test_predictions_match_embedded_reference(self) -> None:
        results = build_results(self.task)
        comparable = [result for result in results['test_results'] if 'passed_reference' in result]
        self.assertTrue(comparable)
        self.assertTrue(all(result['passed_reference'] for result in comparable))

    def test_results_json_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'results.json'
            results = run(results_path=output_path)
            self.assertTrue(output_path.exists())
            saved = json.loads(output_path.read_text())
            self.assertEqual(saved['task_id'], results['task_id'])
            self.assertEqual(saved['training_accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
