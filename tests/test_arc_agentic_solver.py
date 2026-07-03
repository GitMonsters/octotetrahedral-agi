"""Tests for arc_agentic_solver.py."""

import json
import time
from pathlib import Path

import pytest

from arc_agentic_solver import (
    AgenticARCSolver,
    HeuristicCodeProposer,
    LLMCodeProposer,
    NoMoreCandidatesError,
    ProposerUnavailableError,
    UnsafeCodeError,
    _check_code_is_safe,
    run_transform_safely,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "arc-puzzle-catalog" / "dataset" / "tasks"


def load_task(task_id: str) -> dict:
    with open(TASKS_DIR / f"{task_id}.json") as handle:
        return json.load(handle)


# ============================================================================
# Sandbox / safety
# ============================================================================


class TestRunTransformSafely:
    def test_executes_valid_code(self):
        code = "def transform(grid):\n    return grid[::-1]\n"
        outcome = run_transform_safely(code, [[1, 2], [3, 4]])
        assert outcome.success is True
        assert outcome.output == [[3, 4], [1, 2]]
        assert outcome.error is None

    def test_missing_transform_function(self):
        code = "def not_transform(grid):\n    return grid\n"
        outcome = run_transform_safely(code, [[1, 2]])
        assert outcome.success is False
        assert "transform" in outcome.error

    def test_candidate_exception_is_captured(self):
        code = "def transform(grid):\n    return 1 / 0\n"
        outcome = run_transform_safely(code, [[1, 2]])
        assert outcome.success is False
        assert "ZeroDivisionError" in outcome.error

    def test_rejects_import(self):
        code = "import os\ndef transform(grid):\n    return grid\n"
        outcome = run_transform_safely(code, [[1, 2]])
        assert outcome.success is False
        assert "UnsafeCodeError" in outcome.error

    def test_rejects_dunder_escape(self):
        code = "def transform(grid):\n    x = ().__class__.__bases__\n    return grid\n"
        outcome = run_transform_safely(code, [[1, 2]])
        assert outcome.success is False
        assert "UnsafeCodeError" in outcome.error

    def test_rejects_eval(self):
        code = "def transform(grid):\n    return eval('grid')\n"
        outcome = run_transform_safely(code, [[1, 2]])
        assert outcome.success is False
        assert "UnsafeCodeError" in outcome.error

    def test_timeout_on_infinite_loop(self):
        code = "def transform(grid):\n    while True:\n        pass\n"
        start = time.monotonic()
        outcome = run_transform_safely(code, [[1, 2]], timeout=0.3)
        elapsed = time.monotonic() - start
        assert outcome.success is False
        assert "timed out" in outcome.error
        assert elapsed < 5.0

    def test_syntax_error_is_rejected_safely(self):
        code = "def transform(grid:\n    return grid\n"
        outcome = run_transform_safely(code, [[1, 2]])
        assert outcome.success is False
        assert "UnsafeCodeError" in outcome.error


class TestCheckCodeIsSafe:
    def test_allows_plain_code(self):
        _check_code_is_safe("def transform(grid):\n    return [row[::-1] for row in grid]\n")

    @pytest.mark.parametrize(
        "code",
        [
            "import sys\ndef transform(grid):\n    return grid\n",
            "from os import path\ndef transform(grid):\n    return grid\n",
            "def transform(grid):\n    return open('/etc/passwd').read()\n",
            "def transform(grid):\n    return globals()\n",
        ],
    )
    def test_rejects_dangerous_code(self, code):
        with pytest.raises(UnsafeCodeError):
            _check_code_is_safe(code)


# ============================================================================
# HeuristicCodeProposer
# ============================================================================


class TestHeuristicCodeProposer:
    def test_cycles_through_candidates_deterministically(self):
        proposer = HeuristicCodeProposer()
        first_pass = [proposer.propose({}, i) for i in range(7)]
        second_pass = [HeuristicCodeProposer().propose({}, i) for i in range(7)]
        assert first_pass == second_pass

    def test_raises_when_candidates_exhausted(self):
        proposer = HeuristicCodeProposer()
        num_candidates = 0
        while True:
            try:
                proposer.propose({}, num_candidates)
                num_candidates += 1
            except NoMoreCandidatesError:
                break
        with pytest.raises(NoMoreCandidatesError):
            proposer.propose({}, num_candidates)


# ============================================================================
# AgenticARCSolver end-to-end (real ARC tasks, no network)
# ============================================================================


class TestAgenticARCSolverEndToEnd:
    def test_solves_rotate_180_task(self):
        task = load_task("3c9b0459")
        solver = AgenticARCSolver(HeuristicCodeProposer(), max_attempts=10)
        result = solver.solve(task)

        assert result.solved is True
        assert result.solving_code is not None
        assert result.test_predictions == [example["output"] for example in task["test"]]
        assert result.attempts[-1].passed_all_train is True

    def test_solves_flip_h_task(self):
        task = load_task("67a3c6ac")
        solver = AgenticARCSolver(HeuristicCodeProposer(), max_attempts=10)
        result = solver.solve(task)

        assert result.solved is True
        assert result.test_predictions == [example["output"] for example in task["test"]]

    def test_records_failed_attempts_with_feedback_before_success(self):
        task = load_task("3c9b0459")  # solved by rotate_180, the 3rd heuristic candidate
        solver = AgenticARCSolver(HeuristicCodeProposer(), max_attempts=10)
        result = solver.solve(task)

        assert result.solved is True
        failed_attempts = [a for a in result.attempts if not a.passed_all_train]
        assert len(failed_attempts) >= 1
        for attempt in failed_attempts:
            assert any(not r.passed for r in attempt.train_results)

    def test_reports_unsolved_when_heuristics_cannot_match(self):
        # A proposer that only ever offers a single wrong constant-output candidate
        # can never satisfy a real (non-trivial) ARC task's training pairs.
        class AlwaysWrongProposer:
            def propose(self, task, attempt, feedback=None):
                if attempt >= 3:
                    raise NoMoreCandidatesError()
                return "def transform(grid):\n    return [[0]]\n"

        task = load_task("3c9b0459")
        solver = AgenticARCSolver(AlwaysWrongProposer(), max_attempts=10)
        result = solver.solve(task)

        assert result.solved is False
        assert result.test_predictions is None
        assert len(result.attempts) == 3


# ============================================================================
# LLMCodeProposer (mocked HTTP, fully offline)
# ============================================================================


class TestLLMCodeProposer:
    def test_no_api_key_raises_proposer_unavailable(self, monkeypatch):
        monkeypatch.delenv("MERCURY_API_KEY", raising=False)
        monkeypatch.delenv("LLM_FALLBACK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        proposer = LLMCodeProposer()
        with pytest.raises(ProposerUnavailableError):
            proposer.propose({"train": [], "test": []}, 0)

    def test_extracts_code_from_fenced_block(self):
        proposer = LLMCodeProposer(api_key="fake-key")
        text = "Here is the function:\n```python\ndef transform(grid):\n    return grid\n```\nDone."
        assert proposer._extract_code(text) == "def transform(grid):\n    return grid"

    def test_extracts_code_without_fence_as_fallback(self):
        proposer = LLMCodeProposer(api_key="fake-key")
        text = "def transform(grid):\n    return grid\n"
        assert "def transform" in proposer._extract_code(text)

    def test_raises_when_no_transform_function_present(self):
        proposer = LLMCodeProposer(api_key="fake-key")
        with pytest.raises(ProposerUnavailableError):
            proposer._extract_code("I'm not sure how to solve this puzzle.")

    def test_propose_uses_mocked_api_response(self, monkeypatch):
        proposer = LLMCodeProposer(api_key="fake-key")
        monkeypatch.setattr(
            proposer,
            "_call_api",
            lambda prompt: "```python\ndef transform(grid):\n    return grid[::-1]\n```",
        )
        task = {"train": [{"input": [[1]], "output": [[1]]}], "test": []}
        code = proposer.propose(task, 0)
        assert "def transform" in code

        outcome = run_transform_safely(code, [[1, 2], [3, 4]])
        assert outcome.success is True
        assert outcome.output == [[3, 4], [1, 2]]

    def test_propose_includes_feedback_in_prompt(self, monkeypatch):
        proposer = LLMCodeProposer(api_key="fake-key")
        captured_prompts = []

        def fake_call_api(prompt):
            captured_prompts.append(prompt)
            return "```python\ndef transform(grid):\n    return grid\n```"

        monkeypatch.setattr(proposer, "_call_api", fake_call_api)
        task = {"train": [{"input": [[1]], "output": [[1]]}], "test": []}
        proposer.propose(task, 1, feedback="Example 0: expected [[1]] but got [[2]]")

        assert len(captured_prompts) == 1
        assert "Feedback on your previous attempt" in captured_prompts[0]
        assert "expected [[1]] but got [[2]]" in captured_prompts[0]

    def test_propose_raises_when_api_call_fails(self, monkeypatch):
        proposer = LLMCodeProposer(api_key="fake-key")
        monkeypatch.setattr(proposer, "_call_api", lambda prompt: None)
        with pytest.raises(ProposerUnavailableError):
            proposer.propose({"train": [{"input": [[1]], "output": [[1]]}], "test": []}, 0)
