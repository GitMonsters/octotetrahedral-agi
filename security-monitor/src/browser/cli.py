#!/usr/bin/env python3
"""
QuantumShield Secure Browser — Interactive CLI
===============================================
Usage:
    python3 -m browser.cli
    python3 security-monitor/src/browser/cli.py

Commands:
    <url>          Navigate to URL
    [N]            Follow link number N
    back / b       Go back
    forward / f    Go forward
    links / l      Show all links on current page
    scan / s       Show threat report for current page
    status / st    Show trident instance status
    refresh / r    Reload current page
    history / h    Show navigation history
    help / ?       Show this help
    quit / q       Exit browser
"""

import sys
import os
import readline
import textwrap

# Fix imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from browser.trident import TridentBrowser
from browser.sandbox import FetchResult
from browser.scanner import ThreatLevel


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         QuantumShield Secure Browser v1.0                    ║
║         Trident Architecture — 3x Sandboxed Instances        ║
║         Consensus Verification + Malware Scanning            ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
  Commands:
    <url>          Navigate to URL (auto-prepends https:// if needed)
    [N]            Follow link number N from current page
    back / b       Go back in history
    forward / f    Go forward in history
    links / l      Show all links on current page
    scan / s       Show detailed threat report
    status / st    Show all instance status + health
    refresh / r    Reload current page
    history / h    Show navigation history
    help / ?       Show this help
    quit / q       Exit and destroy all sandboxes
"""


def format_status_bar(browser: TridentBrowser) -> str:
    """Format the status bar showing instance health."""
    parts = []
    for i, inst in enumerate(browser.instances):
        name = browser.INSTANCE_NAMES[i]
        if inst.is_alive():
            parts.append(f"  {name}:🟢")
        else:
            parts.append(f"  {name}:🔴")
    alive = sum(1 for inst in browser.instances if inst.is_alive())
    return f"[{alive}/3 alive{' '.join(parts)}] Fetches:{browser.total_fetches} Blocked:{browser.total_blocks}"


def display_result(result: FetchResult, browser: TridentBrowser, width: int = 80) -> None:
    """Display a fetch result in the terminal."""
    # Status line
    if result.error:
        print(f"\n  ❌ Error: {result.error}")
        return

    # Threat indicator
    if result.threat_report:
        print(f"\n  {result.threat_report.summary()}")
        if result.threat_report.blocked:
            print(f"\n  Page blocked for security. Use 'scan' for details.\n")
            return

    # Page content
    print(f"\n  URL: {result.url}")
    print(f"  Status: {result.status_code} | {result.fetch_time_ms:.0f}ms | {len(result.links)} links")
    print(f"  {'─' * (width - 4)}")

    # Word-wrap and display
    lines = result.rendered_text.split("\n")
    for line in lines[:200]:  # Cap at 200 lines
        if len(line) > width - 4:
            wrapped = textwrap.fill(line, width=width - 4)
            for wl in wrapped.split("\n"):
                print(f"  {wl}")
        else:
            print(f"  {line}")

    if len(lines) > 200:
        print(f"\n  ... ({len(lines) - 200} more lines truncated)")

    print(f"  {'─' * (width - 4)}")
    if result.links:
        print(f"  {len(result.links)} links found — type 'links' to see, or [N] to follow")


def display_links(result: FetchResult) -> None:
    """Show all links on the current page."""
    if not result or not result.links:
        print("  No links on current page.")
        return
    print(f"\n  Links ({len(result.links)}):")
    for i, (url, text) in enumerate(result.links):
        text_short = text[:50] + "..." if len(text) > 50 else text
        url_short = url[:60] + "..." if len(url) > 60 else url
        print(f"  [{i}] {text_short}")
        print(f"       → {url_short}")


def display_scan(result: FetchResult) -> None:
    """Show detailed threat scan report."""
    if not result or not result.threat_report:
        print("  No scan data available.")
        return
    r = result.threat_report
    print(f"\n  Threat Scan Report")
    print(f"  {'='*50}")
    print(f"  URL:     {r.url}")
    print(f"  Level:   {r.level.name}")
    print(f"  Hash:    {r.content_hash[:32]}...")
    print(f"  Blocked: {'YES' if r.blocked else 'No'}")
    if r.threats:
        print(f"\n  Threats ({len(r.threats)}):")
        for t in r.threats:
            print(f"    [{t['level']}] {t['type']}")
            print(f"           {t['detail'][:80]}")
    else:
        print(f"\n  ✅ No threats detected")
    print(f"  {'='*50}")


def display_status(browser: TridentBrowser) -> None:
    """Show detailed trident status."""
    st = browser.status()
    print(f"\n  Trident Browser Status")
    print(f"  {'='*50}")
    print(f"  Alive:              {st['alive']}/3")
    print(f"  Total fetches:      {st['total_fetches']}")
    print(f"  Threats blocked:    {st['total_blocks']}")
    print(f"  Consensus failures: {st['consensus_failures']}")
    print(f"  History depth:      {st['history_depth']}")
    print()
    for inst in st["instances"]:
        name = browser.INSTANCE_NAMES[inst["id"]]
        state_icon = "🟢" if inst["state"] not in ("dead", "quarantined") else "🔴"
        print(f"  Instance {name} {state_icon}")
        print(f"    State:    {inst['state']}")
        print(f"    URL:      {inst['url'] or '(none)'}")
        print(f"    Fetches:  {inst['fetches']}")
        print(f"    Blocked:  {inst['blocked']}")
        print(f"    Sandbox:  {inst['sandbox']}")
        print()
    print(f"  {'='*50}")


def display_history(browser: TridentBrowser) -> None:
    """Show navigation history."""
    if not browser.history:
        print("  No history yet.")
        return
    print(f"\n  History ({len(browser.history)} entries):")
    for i, url in enumerate(browser.history):
        marker = " ◀" if i == browser.history_pos else ""
        print(f"  {i}: {url}{marker}")


def main() -> None:
    print(BANNER)
    print("  Initializing trident instances...")
    browser = TridentBrowser(verbose=True)
    print(f"\n  {format_status_bar(browser)}")
    print(f"\n  Type a URL to browse, or 'help' for commands.\n")

    current_result: FetchResult | None = None

    try:
        while True:
            try:
                prompt = f"  🌐 > "
                cmd = input(prompt).strip()
            except EOFError:
                break

            if not cmd:
                continue

            # Commands
            if cmd in ("quit", "q", "exit"):
                break
            elif cmd in ("help", "?"):
                print(HELP_TEXT)
            elif cmd in ("status", "st"):
                display_status(browser)
            elif cmd in ("links", "l"):
                if current_result:
                    display_links(current_result)
                else:
                    print("  No page loaded yet.")
            elif cmd in ("scan", "s"):
                if current_result:
                    display_scan(current_result)
                else:
                    print("  No page loaded yet.")
            elif cmd in ("history", "h"):
                display_history(browser)
            elif cmd in ("refresh", "r"):
                if browser.history and browser.history_pos >= 0:
                    url = browser.history[browser.history_pos]
                    print(f"  Refreshing {url}...")
                    current_result = browser.fetch(url)
                    display_result(current_result, browser)
                else:
                    print("  Nothing to refresh.")
            elif cmd in ("back", "b"):
                url = browser.go_back()
                if url:
                    print(f"  ◀ Back to {url}")
                    current_result = browser.fetch(url)
                    display_result(current_result, browser)
                else:
                    print("  Already at start of history.")
            elif cmd in ("forward", "f"):
                url = browser.go_forward()
                if url:
                    print(f"  ▶ Forward to {url}")
                    current_result = browser.fetch(url)
                    display_result(current_result, browser)
                else:
                    print("  Already at end of history.")
            elif cmd.startswith("[") and cmd.endswith("]"):
                # Follow link
                try:
                    idx = int(cmd[1:-1])
                    if current_result and 0 <= idx < len(current_result.links):
                        url = current_result.links[idx][0]
                        print(f"  Following link [{idx}] → {url}")
                        current_result = browser.fetch(url)
                        display_result(current_result, browser)
                    else:
                        print(f"  Invalid link number: {idx}")
                except ValueError:
                    print(f"  Invalid: {cmd}")
            elif cmd.isdigit():
                # Follow link by number
                idx = int(cmd)
                if current_result and 0 <= idx < len(current_result.links):
                    url = current_result.links[idx][0]
                    print(f"  Following link [{idx}] → {url}")
                    current_result = browser.fetch(url)
                    display_result(current_result, browser)
                else:
                    print(f"  Invalid link number: {idx}")
            else:
                # Treat as URL
                url = cmd
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                print(f"  Fetching through trident ({sum(1 for i in browser.instances if i.is_alive())}/3 instances)...")
                current_result = browser.fetch(url)
                display_result(current_result, browser)

            # Status bar
            print(f"\n  {format_status_bar(browser)}")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        print("\n  Destroying all sandboxes...")
        browser.shutdown()
        print("  QuantumShield Browser exited. All sandboxes destroyed. ✓")


if __name__ == "__main__":
    main()
