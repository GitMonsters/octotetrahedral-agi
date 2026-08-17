#!/usr/bin/env python3
"""ARC synthesis harness CLI.

Examples:
    python -m harness.cli eval --subset v1 --limit 5 --attempts 4
    python -m harness.cli extract
    python -m harness.cli solve 00576224
"""

from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_extract(cfg, args):
    from harness.library import build_library, write_library

    entries = build_library(cfg)
    path = write_library(cfg, entries)
    print(f"Extracted {len(entries)} helper functions -> {path}")


def cmd_eval(cfg, args):
    from harness.evaluate import run_eval

    cfg.max_attempts = args.attempts
    cfg.candidates_per_attempt = args.candidates
    cfg.exclude_solved = args.exclude_solved
    cfg.use_library = not args.no_library
    cfg.num_workers = args.workers
    if args.no_kfold:
        cfg.kfold = False
    if args.no_consensus:
        cfg.consensus = False
    if getattr(args, "initial_candidates", None):
        cfg.initial_candidates = args.initial_candidates
    if args.max_tokens:
        cfg.max_tokens_generate = args.max_tokens
    if getattr(args, "task_timeout", None):
        cfg.task_timeout_s = args.task_timeout
    if getattr(args, "reasoning_effort", None):
        cfg.reasoning_effort = args.reasoning_effort
    if args.synth_model:
        cfg.synth_model = args.synth_model
    if args.candidate_model:
        cfg.candidate_model = args.candidate_model
    report = run_eval(cfg, subset=args.subset, limit=args.limit)
    print(json.dumps(report, indent=2))


def cmd_solve(cfg, args):
    from harness.dataset import load_task
    from harness.synthesizer import Synthesizer
    from harness.verifier import extract_code

    cfg.max_attempts = args.attempts
    cfg.candidates_per_attempt = args.candidates
    task = load_task(cfg, args.task_id)
    synth = Synthesizer(cfg)
    out = synth.solve_task(args.task_id, task)
    print(f"status={out['status']} train_score={out['score']:.2f}", flush=True)
    if out.get("code"):
        path = f"runs/solved_{args.task_id}.py"
        os.makedirs("runs", exist_ok=True)
        with open(path, "w") as f:
            f.write(out["code"])
        print(f"saved -> {path}", flush=True)


def main():
    p = argparse.ArgumentParser(prog="arc-harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extract the helper library from the solver corpus")
    pe.set_defaults(fn=cmd_extract)

    peval = sub.add_parser("eval", help="Run the synthesizer over eval tasks")
    peval.add_argument("--subset", default="v1", choices=["v1", "v2", "train"])
    peval.add_argument("--limit", type=int, default=5)
    peval.add_argument("--attempts", type=int, default=8)
    peval.add_argument("--candidates", type=int, default=2)
    peval.add_argument("--initial-candidates", type=int, default=None)
    peval.add_argument("--synth-model", default=None)
    peval.add_argument("--candidate-model", default=None)
    peval.add_argument("--max-tokens", type=int, default=None)
    peval.add_argument("--task-timeout", type=float, default=None)
    peval.add_argument("--reasoning-effort", default=None)
    peval.add_argument("--exclude-solved", action="store_true")
    peval.add_argument("--no-library", action="store_true")
    peval.add_argument("--no-kfold", action="store_true", help="disable leave-one-out validation")
    peval.add_argument("--no-consensus", action="store_true", help="disable consensus-on-test gate")
    peval.add_argument("--workers", type=int, default=4)
    peval.set_defaults(fn=cmd_eval)

    ps = sub.add_parser("solve", help="Attempt a single task by ID")
    ps.add_argument("task_id")
    ps.add_argument("--attempts", type=int, default=8)
    ps.add_argument("--candidates", type=int, default=2)
    ps.set_defaults(fn=cmd_solve)

    args = p.parse_args()
    from harness.config import HarnessConfig

    cfg = HarnessConfig()
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
