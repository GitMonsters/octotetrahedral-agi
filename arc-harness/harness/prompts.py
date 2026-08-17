"""Prompt construction for ARC program synthesis."""

from __future__ import annotations

import os

from .config import HarnessConfig
from .grids import render_task


SYSTEM_PROMPT = """You are an expert ARC-AGI puzzle solver. ARC tasks are grid puzzles where you must infer a transformation rule from input/output training examples, then apply it to test inputs.

Write a single Python function `solve(grid)` that transforms an input grid into the output grid. Requirements:
- `grid` is a `list[list[int]]` with values 0-9 (0 = black/background).
- Return the output as `list[list[int]]`.
- You may use `numpy` (imported as `np`), `collections`, `itertools`, `copy`, `math`.
- Do NOT read any files, do NOT call `print()`, and keep runtime under a few seconds.
- IMPORTANT: The rule must generalize — do not hardcode specific grids or positions from the training examples. Infer the general transformation.
- Think step by step inside the function comments, then implement the rule.

Respond with ONLY the Python code inside a ```python ... ``` block. No prose before or after."""


def _library_block(cfg: HarnessConfig, helpers: list[dict], library_src: str) -> str:
    """Build a compact block of reusable helpers to include in the prompt."""
    if not helpers:
        return ""
    # Reconstruct source for the selected names from concepts.py
    src_lines = library_src.splitlines()
    selected = []
    for h in helpers:
        name = h["name"]
        # find the function def in the source
        start = None
        for i, line in enumerate(src_lines):
            if line.startswith(f"def {name}(") or line.startswith(f"    def {name}("):
                start = i
                break
        if start is None:
            continue
        # capture until the next top-level def
        j = start + 1
        while j < len(src_lines) and not (
            src_lines[j].startswith("def ") and not src_lines[j].startswith("    ")
        ):
            if src_lines[j].startswith("# ==="):
                break
            j += 1
        block = "\n".join(src_lines[start:j]).rstrip()
        if block:
            selected.append(f"# helper from task {h['task_id']}\n{block}")
    if not selected:
        return ""
    return (
        "The following reusable helper functions are available (from a library of "
        "previously solved ARC tasks). You MAY call them inside solve() if they fit:\n\n"
        + "\n\n".join(selected)
        + "\n\n"
    )


def build_generate_prompt(cfg: HarnessConfig, task: dict, task_id: str, store=None) -> list[dict]:
    """Prompt for the first attempt at a task.

    When a CompoundingStore is provided, injects verified similar solutions
    as few-shot examples and relevant patterns as strategy hints.  Only
    verified (test-passed) results compound into the prompt.
    """
    user = f"Solve ARC task {task_id}. Here are the training examples:\n\n"
    user += render_task(task, include_test=False)
    user += "\n\nWrite solve() that matches ALL training examples and generalizes to the test inputs (which follow the same rule)."

    # compounding: inject verified similar solutions as few-shot examples
    if store is not None:
        features = _task_features_for_prompt(task)
        similar = store.solutions.similar(features, top_k=3)
        if similar:
            block = _verified_solutions_block(similar)
            if block:
                user = block + "\n\n" + user

        # compounding: inject relevant patterns as strategy hints, grouped by MOLT role
        patterns = store.patterns.relevant_to(features, top_k=10)
        if patterns:
            pattern_names = [name for name, score in patterns]
            # group by role
            by_role: dict[str, list[str]] = {}
            for name in pattern_names:
                p = store.patterns._patterns.get(name, {})
                role = p.get("role", "directive")
                by_role.setdefault(role, []).append(name)
            sections = []
            if by_role.get("directive"):
                sections.append(
                    "VERIFIED TRANSFORMS (primary strategies):\n"
                    + ", ".join(by_role["directive"])
                )
            if by_role.get("constraint"):
                sections.append(
                    "CONSTRAINTS (known limits):\n"
                    + ", ".join(by_role["constraint"])
                )
            if by_role.get("heuristic"):
                sections.append(
                    "HEURISTICS (soft guidance):\n"
                    + ", ".join(by_role["heuristic"])
                )
            if by_role.get("context"):
                sections.append(
                    "CONTEXT (background):\n"
                    + ", ".join(by_role["context"])
                )
            if sections:
                pattern_block = (
                    "Previously verified patterns:\n"
                    + "\n".join(sections)
                    + "\n\nConsider whether any of these patterns apply to this task. "
                    "Use them as a starting point but verify they actually fit.\n\n"
                )
                user = pattern_block + user

    if cfg.use_library:
        from .retrieval import LibraryIndex

        idx = LibraryIndex(cfg)
        helpers = idx.top_k_for_task(task_id, cfg.top_k_library)
        lib_path = os.path.join(cfg.library_root, "concepts.py")
        if helpers and os.path.exists(lib_path):
            lib_src = open(lib_path).read()
            block = _library_block(cfg, helpers, lib_src)
            if block:
                user = block + "Task:\n\n" + user

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _task_features_for_prompt(task: dict) -> dict:
    """Extract sparse features for compounding retrieval."""
    grids = [tr["input"] for tr in task.get("train", [])]
    grids += [tr["output"] for tr in task.get("train", [])]
    features = {}
    h_set, w_set, color_set = set(), set(), set()
    for g in grids:
        if g and g[0]:
            h_set.add(len(g))
            w_set.add(len(g[0]))
            for row in g:
                color_set.update(row)
    features["n_colors"] = len(color_set)
    features["max_h"] = max(h_set) if h_set else 0
    features["max_w"] = max(w_set) if w_set else 0
    features["n_train"] = len(task.get("train", []))
    for c in sorted(color_set)[:10]:
        features[f"color_{c}"] = 1.0
    return features


def _verified_solutions_block(solutions: list[dict]) -> str:
    """Build a few-shot block from verified similar solutions."""
    if not solutions:
        return ""
    blocks = []
    for sol in solutions:
        code = sol.get("code", "")
        rules = sol.get("rules", [])
        if not code:
            continue
        header = f"# Verified solution for similar task {sol['task_id']}"
        if rules:
            header += f" (rules: {', '.join(rules)})"
        blocks.append(f"{header}\n{code}")
    if not blocks:
        return ""
    return (
        "The following verified solutions to SIMILAR tasks are provided as examples. "
        "Study their structure and patterns, but DO NOT copy them directly — this task "
        "has different inputs/outputs. Use them to understand the problem-solving approach:\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
    )


def build_refine_prompt(cfg: HarnessConfig, task: dict, task_id: str, code: str, feedback: str, store=None) -> list[dict]:
    """Prompt for refining a solver that failed training examples.

    When a CompoundingStore is provided, injects relevant patterns as
    strategy hints for the refinement.
    """
    user = f"ARC task {task_id}. Your previous solve() failed some training examples. Here is the task again:\n\n"
    user += render_task(task, include_test=False)

    # compounding: inject relevant patterns for refinement guidance, grouped by MOLT role
    if store is not None:
        features = _task_features_for_prompt(task)
        patterns = store.patterns.relevant_to(features, top_k=10)
        if patterns:
            pattern_names = [name for name, score in patterns]
            by_role: dict[str, list[str]] = {}
            for name in pattern_names:
                p = store.patterns._patterns.get(name, {})
                role = p.get("role", "directive")
                by_role.setdefault(role, []).append(name)
            sections = []
            if by_role.get("directive"):
                sections.append("VERIFIED TRANSFORMS: " + ", ".join(by_role["directive"]))
            if by_role.get("constraint"):
                sections.append("CONSTRAINTS: " + ", ".join(by_role["constraint"]))
            if by_role.get("heuristic"):
                sections.append("HEURISTICS: " + ", ".join(by_role["heuristic"]))
            if sections:
                user += (
                    "\n\nRelevant verified patterns:\n"
                    + "\n".join(sections)
                    + "\nConsider whether applying or combining any of these patterns "
                    "would fix the failing cases.\n"
                )

    user += (
        "\n\nYour previous code:\n\n```python\n"
        + code
        + "\n```\n\n"
        + "Verification feedback (your output is compared cell-by-cell with the expected "
        + "output; study the diffs to infer the exact rule you are missing):\n\n"
        + feedback
        + "\n\nDiagnose the transformation rule precisely from the feedback. What did your code "
        + "do differently vs the expected outputs? Fix solve() so it matches ALL training "
        + "examples. Return ONLY the corrected code in a ```python block."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
