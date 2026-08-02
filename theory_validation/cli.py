"""CLI for the AGI theory validation framework.

Commands
--------
report
    Print a status table of every claim in the registry: evidence level,
    external validation level, and whether its code references still
    resolve.

check
    Verify every ``code_ref`` in the registry actually resolves to a real
    module/class. Exits 1 if any claim references code that no longer
    exists (e.g. after a rename/removal) -- i.e. catches claims silently
    becoming stale or overstated as the codebase changes.

Usage examples::

    python -m theory_validation report
    python -m theory_validation report --format json
    python -m theory_validation check
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from theory_validation.claims import CLAIMS, resolve_code_ref

# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    """Print a human- or machine-readable status table of all claims."""
    if args.format == "json":
        payload = [
            {
                **asdict(c),
                "evidence_status": c.evidence_status.value,
                "external_validation": c.external_validation.value,
            }
            for c in CLAIMS
        ]
        print(json.dumps(payload, indent=2))
        return 0

    name_width = 38
    header = f"{'Claim':<{name_width}} {'Evidence':<22} {'External':<16} {'Tests':<6} {'Code OK'}"
    print(header)
    print("-" * len(header))
    untested = 0
    for c in CLAIMS:
        n_tests = len(c.test_refs)
        if c.evidence_status.value == "untested":
            untested += 1
        code_ok = all(resolve_code_ref(r) for r in c.code_refs)
        code_flag = "yes" if code_ok else "STALE"
        display_name = c.name if len(c.name) <= name_width else c.name[: name_width - 1] + "…"
        print(
            f"{display_name:<{name_width}} {c.evidence_status.value:<22} "
            f"{c.external_validation.value:<16} {n_tests:<6} {code_flag}"
        )
    print("-" * len(header))
    print(
        f"{len(CLAIMS)} claims total, {untested} with zero test coverage "
        f"(evidence_status=untested)."
    )
    if untested:
        print(
            "Note: 'untested' means the mechanism is implemented but no test "
            "verifies it does what is claimed -- not that it is broken."
        )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Verify all code_refs resolve; report stale references."""
    stale: list[tuple[str, str]] = []
    for c in CLAIMS:
        for ref in c.code_refs:
            if not resolve_code_ref(ref):
                stale.append((c.id, ref))

    if not stale:
        print(f"OK: all code references across {len(CLAIMS)} claims resolve.")
        return 0

    print(f"FAIL: {len(stale)} stale code reference(s):")
    for claim_id, ref in stale:
        print(f"  [{claim_id}] {ref} does not resolve")
    return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="theory_validation",
        description=(
            "AGI Theory Validation Framework -- tracks evidence for the "
            "repository's cognitive-architecture claims (perception, "
            "symbol grounding, abstraction, causal inference, world "
            "modeling, planning, meta-learning, adaptive plasticity, and "
            "cross-module integration)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m theory_validation report\n"
            "  python -m theory_validation report --format json\n"
            "  python -m theory_validation check\n"
        ),
    )
    sub = root.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    report_p = sub.add_parser(
        "report",
        help="Print the claims status table.",
        description="Display evidence/external-validation status for every registered claim.",
    )
    report_p.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )
    report_p.set_defaults(func=cmd_report)

    check_p = sub.add_parser(
        "check",
        help="Verify all code references still resolve.",
        description="Exit 1 if any claim references code that no longer exists.",
    )
    check_p.set_defaults(func=cmd_check)

    return root


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
