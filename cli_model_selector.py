"""CLI model selector for copilot task commands.

Usage:
    python cli_model_selector.py --model unified-stack --task "explain recursion"
    python cli_model_selector.py --model unified-stack:16-limb --task "plan a route"
    python cli_model_selector.py --list-models

The selected model (and its stats) are displayed in the task output and
persisted to a local state file so subsequent runs reuse the last selection.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from model_registry import ModelRegistry, get_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Backward-compatible default: falls back to gpt-4 when no model is specified
# and no .copilot/config.yml or last_model.json state file is present.
# Projects that want unified-stack by default should set default_model in
# .copilot/config.yml — that value takes precedence over this constant.
DEFAULT_MODEL = "gpt-4"
STATE_FILE = Path(".copilot") / "last_model.json"


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_last_model() -> Optional[str]:
    """Load the last-used model name from the state file."""
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            return data.get("model")
        except (json.JSONDecodeError, OSError):
            pass
    return None


def save_last_model(model_name: str) -> None:
    """Persist the selected model name to the state file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        STATE_FILE.write_text(
            json.dumps({"model": model_name, "saved_at": time.time()}, indent=2)
        )
    except OSError as exc:
        logger.warning("Could not persist model selection: %s", exc)


# ---------------------------------------------------------------------------
# Model stats display
# ---------------------------------------------------------------------------

def format_model_stats(model_name: str, result: Optional[Dict[str, Any]] = None) -> str:
    """Return a human-readable stats block for the selected model."""
    registry = get_registry()
    try:
        meta = registry.get_metadata(model_name)
    except ValueError:
        return f"[model: {model_name}]"

    lines = [
        f"  Model         : {meta.name} — {meta.description}",
        f"  Provider      : {meta.provider}",
        f"  Capabilities  : {', '.join(meta.capabilities) or 'n/a'}",
    ]
    if meta.limbs:
        lines.append(f"  Limbs         : {meta.limbs}")

    if result:
        if "coherence" in result:
            lines.append(f"  Coherence     : {result['coherence']:.4f}")
        if "coupling_strength" in result:
            lines.append(f"  Coupling      : {result['coupling_strength']:.4f}")
        if "action_channel" in result:
            lines.append(f"  Action channel: limb {result['action_channel']}")
        if "latency_ms" in result:
            lines.append(f"  Latency       : {result['latency_ms']:.2f} ms")
        if "limb_states" in result:
            active = sum(1 for v in result["limb_states"] if v > 0.5)
            lines.append(f"  Limbs active  : {active}/{len(result['limb_states'])}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------

def run_task(model_name: str, task: str, verbose: bool = False) -> Dict[str, Any]:
    """Run *task* through the selected model and return a structured result."""
    registry = get_registry()
    canonical = registry.with_fallback(model_name)
    meta = registry.get_metadata(canonical)

    # Encode the task string as an 8/16-limb activation signal
    limb_count = meta.limbs if meta.limbs > 0 else 8
    raw = [hash(f"{task}:{i}") % 1000 / 1000.0 for i in range(limb_count)]

    t0 = time.monotonic()
    model_obj = registry.load(canonical)
    latency_ms = (time.monotonic() - t0) * 1000

    if model_obj is not None and hasattr(model_obj, "forward"):
        t1 = time.monotonic()
        forward_result = model_obj.forward(raw, task_signal=task[:50])
        latency_ms += (time.monotonic() - t1) * 1000
        result: Dict[str, Any] = dict(forward_result)
        result["latency_ms"] = latency_ms
        result["model"] = canonical
        result["task"] = task
    else:
        # External model (gpt-4, claude-3-opus) — simulated response
        result = {
            "model": canonical,
            "task": task,
            "latency_ms": latency_ms,
            "response": f"[{canonical}] Task acknowledged: {task}",
        }

    return result


# ---------------------------------------------------------------------------
# Argument parser factory
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli_model_selector",
        description="Select and run tasks through the unified cognitive stack or external models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_model_selector.py --model unified-stack --task "explain recursion"
  python cli_model_selector.py --model unified-stack:16-limb --task "plan a route"
  python cli_model_selector.py --model gpt-4 --task "summarise this text"
  python cli_model_selector.py --list-models
  python cli_model_selector.py --validate-model unified-stack:16-limb
        """,
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL_SPEC",
        help=(
            "Model to use, e.g. unified-stack, unified-stack:16-limb, gpt-4, claude-3-opus. "
            f"Defaults to last-used model or '{DEFAULT_MODEL}' if no history."
        ),
    )
    parser.add_argument("--task", default=None, metavar="TASK", help="Task description to run.")
    parser.add_argument(
        "--list-models", action="store_true", help="List all registered models and exit."
    )
    parser.add_argument(
        "--validate-model",
        default=None,
        metavar="MODEL_SPEC",
        help="Validate a model specification and print its metadata.",
    )
    parser.add_argument(
        "--show-stats", action="store_true", help="Always display model stats in output."
    )
    parser.add_argument(
        "--json-output", action="store_true", help="Emit structured JSON output."
    )
    parser.add_argument(
        "--no-persist", action="store_false", dest="persist",
        help="Do not save model selection to state file.",
    )
    parser.set_defaults(persist=True)
    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    registry = get_registry()

    # --list-models
    if args.list_models:
        models = registry.list_models()
        if args.json_output:
            print(json.dumps([m.__dict__ for m in models], indent=2))
        else:
            print("Available models:")
            for m in models:
                limb_info = f" ({m.limbs} limbs)" if m.limbs else ""
                print(f"  {m.name:<22} — {m.description}{limb_info}")
        return 0

    # --validate-model
    if args.validate_model:
        try:
            canonical = registry.resolve_name(args.validate_model)
            meta = registry.get_metadata(canonical)
            if args.json_output:
                print(json.dumps(meta.__dict__, indent=2))
            else:
                print(f"✅  '{args.validate_model}' resolves to '{canonical}'")
                print(format_model_stats(canonical))
            return 0
        except ValueError as exc:
            print(f"❌  {exc}", file=sys.stderr)
            return 1

    # Resolve model
    model_spec = args.model or load_last_model() or DEFAULT_MODEL
    try:
        canonical = registry.with_fallback(model_spec)
    except (ValueError, RuntimeError) as exc:
        print(f"❌  {exc}", file=sys.stderr)
        return 1

    if args.persist:
        save_last_model(canonical)

    # No task: just print selected model
    if not args.task:
        if args.json_output:
            print(json.dumps({"model": canonical}))
        else:
            print(f"Selected model: {canonical}")
            if args.show_stats:
                print(format_model_stats(canonical))
        return 0

    # Run task
    result = run_task(canonical, args.task)

    if args.json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'─' * 60}")
        print(f" Task   : {args.task}")
        print(format_model_stats(canonical, result))
        if "response" in result:
            print(f"\n  Response: {result['response']}")
        print(f"{'─' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
