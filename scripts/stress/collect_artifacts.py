"""Stress-test artifact collector.

Writes required metadata files (run_config.json, commit_sha.txt,
environment.txt, summary.md) into the artifact output directory.
Called by the stress shell scripts after the test suite completes.

Usage::

    python scripts/stress/collect_artifacts.py \\
        --out-dir artifacts/stress/quick/20260722T000000Z \\
        --suite quick \\
        --seed 1337 \\
        --command "scripts/stress/quick_suite.sh" \\
        --exit-code 0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_sha() -> str:
    """Return HEAD commit SHA, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return os.environ.get("GITHUB_SHA", "unknown")


def _environment_info() -> str:
    """Collect key environment variables and platform info."""
    lines = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', 'unset')}",
        f"STRESS_SEED={os.environ.get('STRESS_SEED', 'unset')}",
        f"GITHUB_RUN_ID={os.environ.get('GITHUB_RUN_ID', 'local')}",
        f"GITHUB_REF={os.environ.get('GITHUB_REF', 'local')}",
    ]
    return "\n".join(lines)


def collect(out_dir: Path, suite: str, seed: int, command: str, exit_code: int) -> None:
    """Write all required artifact files into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # run_config.json
    run_config = {
        "suite": suite,
        "seed": seed,
        "command": command,
        "exit_code": exit_code,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "unset"),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    # commit_sha.txt
    (out_dir / "commit_sha.txt").write_text(_git_sha() + "\n", encoding="utf-8")

    # environment.txt
    (out_dir / "environment.txt").write_text(_environment_info() + "\n", encoding="utf-8")

    # summary.md (placeholder; populated by individual test scripts)
    summary_path = out_dir / "summary.md"
    if not summary_path.exists():
        status = "PASS" if exit_code == 0 else "FAIL"
        summary_path.write_text(
            f"# Stress Suite: {suite}\n\n"
            f"- **Status:** {status}\n"
            f"- **Exit code:** {exit_code}\n"
            f"- **Command:** `{command}`\n"
            f"- **Seed:** {seed}\n\n"
            "See `run_config.json`, `commit_sha.txt`, and `logs.txt` for details.\n",
            encoding="utf-8",
        )

    print(f"Artifacts written to: {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write stress-test artifact metadata files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for artifacts.")
    parser.add_argument(
        "--suite",
        default="quick",
        choices=["quick", "nightly"],
        help="Suite name (quick or nightly). Default: quick.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Stress seed used. Default: 1337.")
    parser.add_argument("--command", default="", help="Command that was run.")
    parser.add_argument(
        "--exit-code",
        type=int,
        default=0,
        help="Exit code of the stress run. Default: 0.",
    )
    args = parser.parse_args(argv)
    collect(
        out_dir=Path(args.out_dir),
        suite=args.suite,
        seed=args.seed,
        command=args.command,
        exit_code=args.exit_code,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
