"""Sandboxed verification of candidate solve() functions against train pairs."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import textwrap

from .config import HarnessConfig
from .grids import colors_used, grid_dims, render_side_by_side

SOLVER_TEMPLATE = textwrap.dedent(
    """
    import sys, json
    import numpy as np
    from collections import deque, Counter, defaultdict
    import itertools, copy, math

    {code}

    data = json.load(open(sys.argv[1]))
    results = []
    for pair in data:
        try:
            out = solve(pair["input"])
            results.append({{"ok": out == pair["output"], "out": out}})
        except Exception as e:
            results.append({{"ok": False, "error": f"{{type(e).__name__}}: {{e}}"}})
    print(json.dumps(results))
    """
)


def strip_main(code: str) -> str:
    """Remove `if __name__ == "__main__":` blocks so candidate code doesn't
    execute its own harness in the verification subprocess."""
    lines = code.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip().endswith("if __name__ == \"__main__\":"):
            # skip this line and all more-indented following lines
            i += 1
            while i < len(lines) and (lines[i].strip() == "" or lines[i].startswith((" ", "\t"))):
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def extract_code(text: str) -> str:
    """Pull the Python block out of an LLM response."""
    m = re.search(r"```python\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    # no fences: maybe raw code
    if "def solve" in text:
        return text.strip()
    return text.strip()


def _run_candidates(cfg: HarnessConfig, code: str, train_pairs: list[dict]) -> list[dict]:
    """Run solve() against train pairs in a subprocess. Returns per-pair results."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(SOLVER_TEMPLATE.format(code=strip_main(code)))
        solver_path = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(train_pairs, f)
        data_path = f.name

    cmd = [
        sys_executable(),
        solver_path,
        data_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.sandbox_timeout_s,
        )
        if proc.returncode != 0:
            return [{"ok": False, "error": f"process error: {proc.stderr[-500:]}"}]
        out = proc.stdout.strip()
        # find the JSON array (solver prints one line)
        m = re.search(r"\[.*\]", out, re.S)
        if not m:
            return [{"ok": False, "error": f"no result: {out[-300:]}"}]
        return json.loads(m.group(0))
    except subprocess.TimeoutExpired:
        return [{"ok": False, "error": "timeout"}]
    finally:
        import os

        for p in (solver_path, data_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def sys_executable() -> str:
    import sys

    return sys.executable


def _norm_grid(g) -> list[list[int]] | None:
    """Coerce a solver output into a list[list[int]] if possible."""
    if not isinstance(g, list) or not g:
        return None
    if not isinstance(g[0], list):
        return None
    return g


def _observe(pair: dict, got: list[list[int]], exp: list[list[int]]) -> list[str]:
    """High-level observations about what the candidate got wrong."""
    obs = []
    gh, gw = grid_dims(got)
    eh, ew = grid_dims(exp)
    if (gh, gw) != (eh, ew):
        obs.append(f"shape: your output is {gh}x{gw}, expected {eh}x{ew}")
    got_colors, exp_colors = set(colors_used(got)), set(colors_used(exp))
    if got_colors != exp_colors:
        missing = sorted(exp_colors - got_colors)
        extra = sorted(got_colors - exp_colors)
        if missing:
            obs.append(f"missing color(s) {missing}")
        if extra:
            obs.append(f"extra color(s) {extra} (present in output, absent in expected)")
    match, total, mismatches = _cell_stats(got, exp)
    if total:
        obs.append(f"cell match: {match}/{total} ({100 * match // total}%)")
    if mismatches and (gh, gw) == (eh, ew):
        # summarise the top mismatch coordinates in a compact form
        pts = ", ".join(f"({r},{c}){ev}->{gv}" for r, c, ev, gv in mismatches[:12])
        obs.append(f"first differing cells: {pts}")
    return obs


def _cell_stats(got: list[list[int]], exp: list[list[int]]):
    """Count matching/mismatching cells over the union of both grids."""
    gh, gw = grid_dims(got)
    eh, ew = grid_dims(exp)
    total = max(gh, eh) * max(gw, ew)
    match = 0
    mismatches = []
    for r in range(max(gh, eh)):
        for c in range(max(gw, ew)):
            gv = got[r][c] if r < gh and c < gw else None
            ev = exp[r][c] if r < eh and c < ew else None
            if gv == ev:
                match += 1
            elif len(mismatches) < 20:
                mismatches.append((r, c, ev, gv))
    return match, total, mismatches


def _build_feedback(passed: int, total: int, results: list[dict], task: dict) -> str:
    """Build a visual, cell-level feedback string for refinement."""
    lines = [f"Training match: {passed}/{total} examples."]
    if passed == total:
        lines.append("All training examples pass — the rule generalizes.")
        return "\n".join(lines)

    # per-example one-liners
    for i, (r, pair) in enumerate(zip(results, task["train"])):
        if r.get("ok"):
            lines.append(f"Example {i}: PASS")
        elif r.get("error"):
            lines.append(f"Example {i}: RUNTIME ERROR -> {r['error']}")
        else:
            got = _norm_grid(r.get("out"))
            exp = _norm_grid(pair["output"])
            if got is None:
                lines.append(f"Example {i}: returned a non-grid value ({type(r.get('out')).__name__})")
                continue
            gh, gw = grid_dims(got)
            eh, ew = grid_dims(exp)
            if (gh, gw) != (eh, ew):
                lines.append(f"Example {i}: FAIL - wrong shape (got {gh}x{gw}, expected {eh}x{ew})")
            else:
                match, _, _ = _cell_stats(got, exp)
                lines.append(f"Example {i}: FAIL - {match}/{gw * ew} cells correct")

    # deep-dive on the FIRST failing example with a visual diff
    for i, (r, pair) in enumerate(zip(results, task["train"])):
        if r.get("ok") or r.get("error"):
            continue
        got = _norm_grid(r.get("out"))
        if got is None:
            continue
        exp = pair["output"]
        lines.append(f"\nDeep dive on Example {i}:")
        gh, gw = grid_dims(pair["input"])
        if gh <= 16 and gw <= 16:
            lines.append("Input -> (your output) vs (expected), same row alignment:")
            lines.append(render_side_by_side(pair["input"], got, exp))
        else:
            lines.append(f"(grid is {gh}x{gw}, too large to render row-by-row)")
        for o in _observe(pair, got, exp):
            lines.append(f"- {o}")
        # brief mention of remaining failures without dumping them all
        more = sum(
            1 for j, (rj, pj) in enumerate(zip(results, task["train"]))
            if j > i and not rj.get("ok") and not rj.get("error")
        )
        if more:
            lines.append(f"\n(… {more} other failing example(s); fix this one's rule and re-verify.)")
        break
    return "\n".join(lines)


def verify_code(cfg: HarnessConfig, code: str, task: dict) -> dict:
    """Verify code against ALL train pairs. Returns score (cell accuracy) + feedback."""
    train_pairs = task["train"]
    if not train_pairs:
        return {"score": 0.0, "passed": 0, "total": 0, "feedback": "no train pairs"}

    results = _run_candidates(cfg, code, train_pairs)
    passed = sum(1 for r in results if r.get("ok"))
    total = len(results)

    # cell-level accuracy over all pairs (finer signal for the refine loop)
    cell_hits = 0
    cell_total = 0
    for r, pair in zip(results, train_pairs):
        if r.get("ok"):
            m, t, _ = _cell_stats(pair["output"], pair["output"])
            cell_hits += m
            cell_total += t
        elif not r.get("error"):
            got = _norm_grid(r.get("out"))
            if got is not None:
                m, t, _ = _cell_stats(got, pair["output"])
                cell_hits += m
                cell_total += t
    score = (cell_hits / cell_total) if cell_total else 0.0

    feedback = _build_feedback(passed, total, results, task)
    return {
        "score": score,
        "passed": passed,
        "total": total,
        "feedback": feedback,
        "results": results,
    }
