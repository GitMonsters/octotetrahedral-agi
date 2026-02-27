#!/usr/bin/env python3
"""
Computer Agent CLI — run computer-use tasks from the command line.

Usage:
    python -m computer_agent "Open Safari and search for ARC Prize"
    python -m computer_agent --provider openai --confirm "Create a new folder on Desktop"
    python -m computer_agent --interactive
"""

import sys
import os
import argparse
import json
from computer_agent.agent import ComputerAgent
from computer_agent.config import AgentConfig


def main():
    parser = argparse.ArgumentParser(
        description="Computer Agent — AI-powered computer use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m computer_agent "Open Terminal and run ls"
  python -m computer_agent --provider openai --model gpt-4o "Search Google for ARC Prize"
  python -m computer_agent --confirm "Delete old files from Downloads"
  python -m computer_agent --interactive
  python -m computer_agent --replay /tmp/session.json

Environment Variables:
  OPENAI_API_KEY      — OpenAI API key (for GPT-4o vision)
  MERCURY_API_KEY     — Mercury 2 API key
  LLM_API_KEY         — Generic fallback API key
        """
    )

    parser.add_argument("task", nargs="?", help="Task to execute")
    parser.add_argument("--provider", default="openai",
                       choices=["openai", "anthropic", "mercury"],
                       help="LLM provider (default: openai)")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--step-delay", type=float, default=1.0)
    parser.add_argument("--confirm", action="store_true",
                       help="Require confirmation before each action")
    parser.add_argument("--safe-mode", action="store_true",
                       help="Restrict to bounded region only")
    parser.add_argument("--bounds", type=str, default=None,
                       help="Safety bounds: x,y,w,h")
    parser.add_argument("--scale", type=float, default=1.0,
                       help="Screenshot downscale factor (0.5 = half res)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode: enter tasks one at a time")
    parser.add_argument("--replay", type=str, default=None,
                       help="Replay a saved session file")
    parser.add_argument("--save-session", type=str, default=None,
                       help="Save session to file after completion")
    parser.add_argument("--screenshot-only", action="store_true",
                       help="Just take a screenshot and exit")

    args = parser.parse_args()

    # Screenshot-only mode
    if args.screenshot_only:
        from computer_agent.screen import ScreenCapture
        sc = ScreenCapture()
        b64, path = sc.capture()
        print(f"Screenshot saved: {path}")
        print(f"Screen size: {sc.get_screen_size()}")
        return

    # Parse bounds
    bounds = None
    if args.bounds:
        parts = [int(x) for x in args.bounds.split(",")]
        if len(parts) == 4:
            bounds = tuple(parts)

    # Build config
    config = AgentConfig(
        api_key=args.api_key,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        max_steps=args.max_steps,
        step_delay=args.step_delay,
        confirm_actions=args.confirm,
        safe_mode=args.safe_mode,
        bounds=bounds,
        screenshot_scale=args.scale,
        verbose=not args.quiet,
        temperature=args.temperature,
    )

    agent = ComputerAgent(config=config)

    # Replay mode
    if args.replay:
        print(f"Replaying session: {args.replay}")
        agent.replay(args.replay)
        return

    # Interactive mode
    if args.interactive:
        print("Computer Agent — Interactive Mode")
        print("Type a task and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                task = input("Task> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not task or task.lower() in ("quit", "exit", "q"):
                break
            result = agent.run(task)
            print(f"\nResult: {result['status']} — {result['message']}")
            print(f"Steps: {result['steps']}\n")
        return

    # Single task mode
    if not args.task:
        parser.print_help()
        print("\nError: No task specified. Use --interactive for interactive mode.")
        sys.exit(1)

    result = agent.run(args.task, max_steps=args.max_steps)

    # Save session if requested
    if args.save_session:
        agent.save_session(args.save_session)
        print(f"Session saved: {args.save_session}")

    # Print result
    print(f"\n{'='*40}")
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Steps: {result['steps']}")
    print(f"{'='*40}")

    sys.exit(0 if result["status"] == "done" else 1)


if __name__ == "__main__":
    main()
