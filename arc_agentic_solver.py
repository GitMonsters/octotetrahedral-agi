#!/usr/bin/env python3
"""Agentic ARC-AGI solver: propose code, execute it, verify, retry.

Modeled on Microsoft's rStar2-Agent (arXiv:2508.20722), whose core insight is
that a model should verify its own reasoning by writing and *executing* code
rather than just emitting a longer chain-of-thought. Applied to ARC-AGI here:

    1. A `CodeProposer` writes a `transform(grid)` Python function for the task.
    2. The candidate is executed (in a restricted, process-isolated sandbox)
       against every training input/output pair.
    3. If it does not match all training pairs exactly, the mismatch/error is
       fed back to the proposer as context for the next attempt.
    4. Once a candidate passes every training pair, it is applied to the test
       input(s) and returned as the verified prediction.

This differs from `arc_solver.py`'s `LLMFallbackSolver`, which asks a model to
predict the output grid directly (no verification, no retry-on-failure loop).

Two `CodeProposer` implementations are provided:
    - `HeuristicCodeProposer`: cycles through basic geometric transforms with
      no network access. Useful offline/in tests, and as a default demo.
    - `LLMCodeProposer`: calls an OpenAI-compatible chat completion API
      (same provider/env-var conventions as `arc_solver.py`'s
      `LLMFallbackSolver`) to write arbitrary candidate code.

Security note: `run_transform_safely` statically rejects imports, dunder
attribute access, and other obviously dangerous constructs, then executes the
candidate in a separate, timeboxed subprocess. This is a best-effort, defense
-in-depth sandbox suitable for local development — it is not a substitute for
real containerization (as rStar2-Agent's separate "Code Judge" service does)
before pointing this at fully untrusted models in production.
"""

from __future__ import annotations

import ast
import json
import multiprocessing
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

Grid = list[list[int]]


class UnsafeCodeError(ValueError):
    """Raised when proposed code fails the static safety check."""


class ProposerExhaustedError(Exception):
    """Raised by a CodeProposer when it cannot produce any further candidates."""


class NoMoreCandidatesError(ProposerExhaustedError):
    """The heuristic proposer has cycled through all of its candidates."""


class ProposerUnavailableError(ProposerExhaustedError):
    """The LLM proposer could not reach its backing API (e.g. missing key)."""


# ============================================================================
# Sandboxed execution
# ============================================================================

_FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "exit", "quit", "help", "breakpoint",
}


def _check_code_is_safe(code: str) -> None:
    """Best-effort static AST check rejecting obviously dangerous constructs.

    Candidate code may only use plain Python builtins on the provided grid:
    no imports, no file/network access, no dunder attribute access (blocks
    common sandbox-escape gadgets like `"".__class__.__bases__`).
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise UnsafeCodeError(f"code does not parse: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("imports are not allowed in candidate code")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"use of '{node.id}' is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise UnsafeCodeError(f"dunder attribute access '{node.attr}' is not allowed")


def _sandbox_worker(code: str, input_grid: Grid, result_queue: multiprocessing.Queue) -> None:
    """Runs in an isolated child process: exec the candidate and call transform(grid)."""
    import builtins

    safe_names = (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "map", "max", "min", "range", "reversed", "round",
        "set", "sorted", "str", "sum", "tuple", "zip", "isinstance",
    )
    safe_builtins = {name: getattr(builtins, name) for name in safe_names}
    namespace: dict = {"__builtins__": safe_builtins}

    try:
        exec(code, namespace)  # noqa: S102 - isolated subprocess, restricted builtins, pre-vetted AST
        transform = namespace.get("transform")
        if not callable(transform):
            result_queue.put((False, None, "code did not define a callable 'transform(grid)'"))
            return
        output = transform(input_grid)
        if hasattr(output, "tolist"):
            output = output.tolist()
        result_queue.put((True, output, None))
    except Exception as exc:  # noqa: BLE001 - report any candidate-code failure back to the caller
        result_queue.put((False, None, f"{type(exc).__name__}: {exc}"))


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """Result of executing one candidate `transform(grid)` on one input grid."""

    success: bool
    output: Grid | None
    error: str | None


def run_transform_safely(code: str, input_grid: Grid, timeout: float = 2.0) -> ExecutionOutcome:
    """Execute `code`'s `transform(grid)` against `input_grid` in an isolated,
    timeboxed subprocess. Always returns an `ExecutionOutcome` — never raises.
    """
    try:
        _check_code_is_safe(code)
    except UnsafeCodeError as exc:
        return ExecutionOutcome(success=False, output=None, error=f"UnsafeCodeError: {exc}")

    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    process = ctx.Process(target=_sandbox_worker, args=(code, input_grid, result_queue))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return ExecutionOutcome(success=False, output=None, error=f"execution timed out after {timeout}s")

    if not result_queue.empty():
        success, output, error = result_queue.get()
        return ExecutionOutcome(success=success, output=output, error=error)

    return ExecutionOutcome(
        success=False, output=None, error=f"process exited with code {process.exitcode} and no result"
    )


# ============================================================================
# Code proposers
# ============================================================================


class CodeProposer(ABC):
    """Generates candidate `transform(grid)` source code for an ARC task."""

    @abstractmethod
    def propose(self, task: dict, attempt: int, feedback: str | None = None) -> str:
        """Return Python source defining `transform(grid: list[list[int]]) -> list[list[int]]`.

        `attempt` is a 0-based counter; `feedback` describes why the previous
        attempt (if any) failed, so the proposer can self-correct.
        """


_PRIMITIVE_CANDIDATES: dict[str, str] = {
    "identity": "def transform(grid):\n    return grid\n",
    "rotate_90": "def transform(grid):\n    return [list(row) for row in zip(*grid[::-1])]\n",
    "rotate_180": "def transform(grid):\n    return [row[::-1] for row in grid[::-1]]\n",
    "rotate_270": "def transform(grid):\n    return [list(row) for row in zip(*grid)][::-1]\n",
    "flip_h": "def transform(grid):\n    return [row[::-1] for row in grid]\n",
    "flip_v": "def transform(grid):\n    return grid[::-1]\n",
    "transpose": "def transform(grid):\n    return [list(row) for row in zip(*grid)]\n",
}


class HeuristicCodeProposer(CodeProposer):
    """Deterministic, offline candidate generator cycling through basic
    geometric transforms. No network access required — useful as a
    default/demo proposer and in tests. Real coverage of arbitrary ARC tasks
    requires `LLMCodeProposer` (or one of this repo's many dedicated
    solvers, e.g. `arc_solver.py`).
    """

    def __init__(self) -> None:
        self._candidate_names = list(_PRIMITIVE_CANDIDATES)

    def propose(self, task: dict, attempt: int, feedback: str | None = None) -> str:
        if attempt >= len(self._candidate_names):
            raise NoMoreCandidatesError("no more heuristic candidates to try")
        return _PRIMITIVE_CANDIDATES[self._candidate_names[attempt]]


_CODE_FENCE_PATTERN = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


class LLMCodeProposer(CodeProposer):
    """Calls an OpenAI-compatible chat completion API to write candidate code.

    Mirrors `arc_solver.py`'s `LLMFallbackSolver` conventions (provider
    presets, env var lookup, stdlib-only HTTP) so existing API key
    configuration works for both solvers.
    """

    PROVIDERS = {
        "mercury": {"base_url": "https://api.inceptionlabs.ai/v1", "default_model": "mercury-coder-small"},
        "openai": {"base_url": "https://api.openai.com/v1", "default_model": "gpt-4o-mini"},
        "anthropic_openai": {"base_url": "https://api.anthropic.com/v1", "default_model": "claude-sonnet-4-20250514"},
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        provider: str = "mercury",
        temperature: float = 0.2,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        import os

        prov = self.PROVIDERS.get(provider, self.PROVIDERS["mercury"])
        self.base_url = base_url or prov["base_url"]
        self.model = model or prov["default_model"]
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = (
            api_key
            or os.environ.get("MERCURY_API_KEY")
            or os.environ.get("LLM_FALLBACK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

    def _format_grid(self, grid: Grid) -> str:
        return "\n".join(" ".join(str(v) for v in row) for row in grid)

    def _build_prompt(self, task: dict, attempt: int, feedback: str | None) -> str:
        parts = [
            "You are solving an ARC-AGI puzzle by writing a Python function.",
            "Write a function `transform(grid)` that maps an input grid to the output grid.",
            "`grid` is a list of lists of integers (0-9 color codes). Return a new list of lists.",
            "Use only plain Python (loops, comprehensions, indexing/slicing) - no imports, no file or network access.",
            "Return ONLY a single fenced python code block containing the function definition.\n",
        ]
        for i, example in enumerate(task["train"]):
            parts.append(f"--- Training Example {i + 1} ---")
            parts.append(f"Input ({len(example['input'])}x{len(example['input'][0])}):")
            parts.append(self._format_grid(example["input"]))
            parts.append(f"Output ({len(example['output'])}x{len(example['output'][0])}):")
            parts.append(self._format_grid(example["output"]))
            parts.append("")

        if feedback:
            parts.append(f"--- Feedback on your previous attempt (#{attempt}) ---")
            parts.append(feedback)
            parts.append("Please fix the function and try again.\n")

        return "\n".join(parts)

    def _extract_code(self, text: str) -> str:
        match = _CODE_FENCE_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        if "def transform" in text:
            return text.strip()
        raise ProposerUnavailableError("LLM response did not contain a 'transform' function")

    def _call_api(self, prompt: str) -> str | None:
        import urllib.error
        import urllib.request

        if not self.api_key:
            return None

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert Python programmer solving ARC-AGI puzzles by writing code.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": 2048,
            }
        ).encode("utf-8")

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError, TimeoutError):
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return None
        return None

    def propose(self, task: dict, attempt: int, feedback: str | None = None) -> str:
        if not self.api_key:
            raise ProposerUnavailableError(
                "No API key configured. Set MERCURY_API_KEY, LLM_FALLBACK_API_KEY, or OPENAI_API_KEY."
            )
        prompt = self._build_prompt(task, attempt, feedback)
        response = self._call_api(prompt)
        if response is None:
            raise ProposerUnavailableError(f"LLM API call to {self.base_url} failed")
        return self._extract_code(response)


# ============================================================================
# Agentic solve loop
# ============================================================================


@dataclass(frozen=True, slots=True)
class TrainPairResult:
    index: int
    passed: bool
    error: str | None
    expected: Grid
    actual: Grid | None


@dataclass(frozen=True, slots=True)
class AgenticAttempt:
    attempt_number: int
    code: str
    train_results: list[TrainPairResult]
    passed_all_train: bool


@dataclass(frozen=True, slots=True)
class AgenticSolveResult:
    solved: bool
    attempts: list[AgenticAttempt] = field(default_factory=list)
    solving_code: str | None = None
    test_predictions: list[Grid | None] | None = None


class AgenticARCSolver:
    """Propose -> execute -> verify -> retry loop for ARC-AGI tasks."""

    def __init__(self, proposer: CodeProposer, max_attempts: int = 5, execution_timeout: float = 2.0) -> None:
        self.proposer = proposer
        self.max_attempts = max_attempts
        self.execution_timeout = execution_timeout

    def solve(self, task: dict) -> AgenticSolveResult:
        attempts: list[AgenticAttempt] = []
        feedback: str | None = None

        for attempt_number in range(self.max_attempts):
            try:
                code = self.proposer.propose(task, attempt_number, feedback)
            except ProposerExhaustedError:
                break

            train_results = [
                self._check_train_pair(code, index, example)
                for index, example in enumerate(task["train"])
            ]
            passed_all = all(result.passed for result in train_results)
            attempts.append(AgenticAttempt(attempt_number, code, train_results, passed_all))

            if passed_all:
                test_predictions = [
                    run_transform_safely(code, example["input"], timeout=self.execution_timeout).output
                    for example in task["test"]
                ]
                return AgenticSolveResult(
                    solved=True, attempts=attempts, solving_code=code, test_predictions=test_predictions
                )

            feedback = self._build_feedback(train_results)

        return AgenticSolveResult(solved=False, attempts=attempts)

    def _check_train_pair(self, code: str, index: int, example: dict) -> TrainPairResult:
        outcome = run_transform_safely(code, example["input"], timeout=self.execution_timeout)
        passed = outcome.success and outcome.output == example["output"]
        return TrainPairResult(
            index=index, passed=passed, error=outcome.error, expected=example["output"], actual=outcome.output
        )

    @staticmethod
    def _build_feedback(train_results: list[TrainPairResult]) -> str:
        failures = [result for result in train_results if not result.passed]
        lines = [f"{len(failures)}/{len(train_results)} training examples failed."]
        first = failures[0]
        if first.error:
            lines.append(f"Example {first.index}: raised {first.error}")
        else:
            lines.append(f"Example {first.index}: expected {first.expected} but got {first.actual}")
        return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================


def _load_task(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Agentic ARC-AGI solver (propose code, execute, verify, retry).")
    parser.add_argument("task", type=str, help="Path to an ARC task JSON file.")
    parser.add_argument(
        "--provider",
        type=str,
        default="heuristic",
        choices=["heuristic", "mercury", "openai", "anthropic_openai"],
        help="Code proposer to use (default: heuristic, no API key required).",
    )
    parser.add_argument("--max-attempts", type=int, default=5, help="Max propose/verify attempts (default: 5).")
    args = parser.parse_args(argv)

    task = _load_task(args.task)
    proposer: CodeProposer
    if args.provider == "heuristic":
        proposer = HeuristicCodeProposer()
    else:
        proposer = LLMCodeProposer(provider=args.provider)

    solver = AgenticARCSolver(proposer, max_attempts=args.max_attempts)
    result = solver.solve(task)

    print(f"Attempts: {len(result.attempts)}")
    if result.solved:
        print("Solved! Verified code:")
        print(result.solving_code)
        print("Test predictions:", result.test_predictions)
    else:
        print("Not solved within attempt budget.")
        if result.attempts:
            print("Last feedback:", AgenticARCSolver._build_feedback(result.attempts[-1].train_results))


if __name__ == "__main__":
    main()
