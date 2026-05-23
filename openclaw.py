#!/usr/bin/env python3
"""
openclaw.py — Parallel Lobster Execution Harness for ARC strategies.

8 arms, each cooking a lobster a different way, all at once.
First arm that succeeds wins. Traces captured for every arm.

Usage (matches the lobster script):

    from openclaw import OpenClawHarness, cooking_methods
    result = await OpenClawHarness.execute_task(task=arc_task, reasoning_trace=True)

Or as a drop-in Synthesizer replacement:

    from openclaw import OpenClawSynth
    syn = OpenClawSynth()
    prog = syn.synthesize(pairs)
"""

import asyncio
import concurrent.futures
import sys
import time
from typing import List, Optional, Tuple

sys.path.insert(0, '/Users/evanpieser')
from dsl.synthesizer import _STRATEGIES

# ── 8 arms: partition 68 strategies into 8 groups by cooking style ─────────

cooking_methods = [
    "steaming",    # geometric / identity  — pure shape transforms
    "boiling",     # color remapping        — color-only transforms
    "grilling",    # object manipulation    — component / recolor ops
    "poaching",    # region fills           — flood / enclosed / border
    "sous-vide",   # tiling & scaling       — tile / scale / fractal
    "baking",      # symmetry & completion  — symmetry / rotation / mirror
    "deep-frying", # line & connectivity    — connect / extend / slide
    "smoking",     # brute-force & complex  — stamp / enumerate / depth search
]

_ARM_STRATEGY_NAMES = [
    # Arm 0 — steaming: geometric / identity
    ["identity", "geometric_transform", "crop_content", "extract_region",
     "variable_pixel_scale", "block_reduce", "scale_up"],

    # Arm 1 — boiling: color ops
    ["color_mapping", "invert_tile", "adjacent_recolor", "color_key_table",
     "complement_recolor", "sorted_color_cycle", "uniform_output"],

    # Arm 2 — grilling: object manipulation
    ["small_component_recolor", "recolor_by_object_size", "shape_match_recolor",
     "largest_object_extract", "fill_bbox_objects", "object_copy_to_marker",
     "neighbor_count_recolor"],

    # Arm 3 — poaching: region fills
    ["fill_enclosed", "interior_fill", "frame_fill", "region_boolean",
     "fill_border_nonbg", "complete_rect_outline", "reverse_concentric"],

    # Arm 4 — sous-vide: tiling & scaling
    ["tiling", "checkerboard_tile", "fractal_self_multiply", "diagonal_tile",
     "stripe_tiling", "block_tile_down", "checkerboard_extend"],

    # Arm 5 — baking: symmetry & rotation
    ["complete_symmetry", "sym_complete_rot180", "rotation_4fold", "mirror_4fold",
     "ring_color_rotate", "row_col_intersect_mark", "color_decoration"],

    # Arm 6 — deep-frying: lines & connectivity
    ["connect_diagonal", "extend_lines", "extend_line_to_border", "diagonal_extend",
     "connect_same_color_lines", "connect_same_color_pairs", "connect_pair_dots",
     "slide_to_border", "gravity_down", "gravity_up", "gravity_left", "gravity_right",
     "gravity_toward_object", "shift_content", "color_slide_direction"],

    # Arm 7 — smoking: brute-force & composite
    ["stamp_shape_at_marker", "stamp_by_mapping", "separator_template",
     "separator_subgrid_reduce", "run_length_group", "dot_row_zones",
     "color_by_proximity", "plus_expand", "rectangle_corner_mark",
     "histogram_barchart", "column_height_rank", "row_height_rank",
     "enumerate_depth1"],
]

# Build lookup: name → (fn, max_t)
_STRAT_LOOKUP = {name: (fn, mt) for name, fn, mt in _STRATEGIES}

# Each arm: list of (name, fn, max_t) for its assigned strategies
ARMS: List[List[Tuple]] = []
_assigned = set()
for arm_names in _ARM_STRATEGY_NAMES:
    arm = [(n, *_STRAT_LOOKUP[n]) for n in arm_names if n in _STRAT_LOOKUP]
    _assigned.update(n for n, *_ in arm)
    ARMS.append(arm)

# Overflow arm 0 catches any unassigned strategies
for name, fn, mt in _STRATEGIES:
    if name not in _assigned:
        ARMS[0].append((name, fn, mt))


# ── Per-arm worker (runs in thread pool) ────────────────────────────────────

def _arm_worker(arm_id: int, method: str, pairs: list, time_budget: float) -> dict:
    """One lobster, one cooking method. Returns trace dict."""
    steps = []
    winner = None
    winner_prog = None
    t0 = time.time()
    remaining = time_budget

    for name, fn, max_t in ARMS[arm_id]:
        if remaining <= 0:
            break
        t_step = time.time()
        prog = None
        try:
            prog = fn(pairs)
        except Exception:
            pass
        elapsed_ms = (time.time() - t_step) * 1000
        remaining = time_budget - (time.time() - t0)

        if prog is not None:
            steps.append({"strategy": name, "outcome": "success",
                          "time_ms": round(elapsed_ms, 2)})
            winner = name
            winner_prog = prog
            break
        else:
            steps.append({"strategy": name, "outcome": "no_program",
                          "time_ms": round(elapsed_ms, 2)})

    return {
        "arm_id": arm_id,
        "method": method,
        "winner": winner,
        "prog": winner_prog,
        "steps": steps,
        "total_ms": round((time.time() - t0) * 1000, 1),
        "status": f"✓ {winner}" if winner else "✗ no solution",
    }


# ── OpenClawHarness ──────────────────────────────────────────────────────────

class OpenClawHarness:
    """
    Parallel strategy executor.  All 8 arms cook simultaneously;
    the first arm that finds a valid program wins.
    """

    @staticmethod
    async def execute_task(
        task: dict,
        reasoning_trace: bool = True,
        time_budget: float = 5.0,
    ) -> dict:
        """
        Cook one ARC task with all 8 arms in parallel.
        Returns a result dict with 'winner', 'prog', 'arm_results', 'status'.
        """
        pairs = [(ex["input"], ex["output"]) for ex in task.get("train", [])]
        loop = asyncio.get_event_loop()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                loop.run_in_executor(
                    pool, _arm_worker, arm_id, cooking_methods[arm_id], pairs, time_budget
                )
                for arm_id in range(8)
            ]
            arm_results = await asyncio.gather(*futures)

        # First arm (by arm_id order) that won
        winner_result = next((r for r in arm_results if r["winner"]), None)

        predictions = []
        if winner_result and winner_result["prog"]:
            prog = winner_result["prog"]
            for tc in task.get("test", []):
                try:
                    predictions.append(prog(tc["input"]))
                except Exception:
                    predictions.append(None)

        result = {
            "winner": winner_result["winner"] if winner_result else None,
            "winning_arm": winner_result["arm_id"] if winner_result else None,
            "winning_method": winner_result["method"] if winner_result else None,
            "predictions": predictions,
            "arm_results": [
                {"arm": r["arm_id"], "method": r["method"],
                 "status": r["status"], "time_ms": r["total_ms"]}
                for r in arm_results
            ],
            "status": winner_result["status"] if winner_result else "✗ all arms failed",
        }

        if reasoning_trace:
            result["trace"] = {
                "arms": arm_results,
                "winner_arm": winner_result["arm_id"] if winner_result else None,
            }

        return result

    @staticmethod
    def solve_sync(task: dict, time_budget: float = 5.0) -> Optional[list]:
        """Synchronous wrapper — drop-in replacement for Synthesizer.solve_task()."""
        return asyncio.run(OpenClawHarness._solve_sync_inner(task, time_budget))

    @staticmethod
    async def _solve_sync_inner(task, time_budget):
        result = await OpenClawHarness.execute_task(task, reasoning_trace=False,
                                                     time_budget=time_budget)
        preds = result.get("predictions", [])
        return preds[0] if preds else None


# ── OpenClawSynth — drop-in Synthesizer replacement ─────────────────────────

class OpenClawSynth:
    """
    Drop-in replacement for dsl.synthesizer.Synthesizer.
    Uses 8 parallel arms instead of sequential strategy scan.
    """

    def __init__(self, time_budget: float = 5.0):
        self.time_budget = time_budget

    def synthesize(self, pairs, verbose: bool = False):
        """Find a program using 8 parallel arms. Returns first winning program."""
        loop = asyncio.new_event_loop()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [
                    loop.run_in_executor(pool, _arm_worker, arm_id,
                                         cooking_methods[arm_id], pairs,
                                         self.time_budget)
                    for arm_id in range(8)
                ]
                arm_results = loop.run_until_complete(asyncio.gather(*futures))
        finally:
            loop.close()

        winner = next((r for r in arm_results if r["winner"]), None)
        if verbose and winner:
            print(f"  [OpenClaw] ✓ arm {winner['arm_id']} ({winner['method']}) "
                  f"→ {winner['winner']} ({winner['total_ms']:.1f} ms)")
        elif verbose:
            print("  [OpenClaw] ✗ all 8 arms failed")
        return winner["prog"] if winner else None

    def solve_task(self, task, verbose=False):
        pairs = [(ex["input"], ex["output"]) for ex in task.get("train", [])]
        test_input = task["test"][0]["input"]
        prog = self.synthesize(pairs, verbose=verbose)
        if prog is None:
            return None
        try:
            return prog(test_input)
        except Exception:
            return None


# ── Standalone demo (the original lobster script, for real) ─────────────────

async def _demo():
    import json
    path = '/Users/evanpieser/Downloads/re-arc_test_challenges-2026-05-23T07-46-05.json'
    challenges = json.load(open(path))
    task_ids = list(challenges.keys())[:8]

    print("🦞 Parallel Lobster Execution — 8 arms, 8 methods, 8 tasks\n")

    async def cook_lobster(arm_id, method, tid):
        task = challenges[tid]
        print(f"  [Arm {arm_id}] Starting {method} on {tid}…")
        result = await OpenClawHarness.execute_task(task, reasoning_trace=False,
                                                    time_budget=3.0)
        return f"  Arm {arm_id} ({method}) [{tid}]: {result['status']}"

    coros = [cook_lobster(i, cooking_methods[i], tid)
             for i, tid in enumerate(task_ids)]
    results = await asyncio.gather(*coros)
    print()
    for r in results:
        print(r)

if __name__ == "__main__":
    asyncio.run(_demo())
