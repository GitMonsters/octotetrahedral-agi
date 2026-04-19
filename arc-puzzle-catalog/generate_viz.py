#!/usr/bin/env python3
"""Generate HTML grid visualizations for ARC-AGI tasks."""

import json
import os
from pathlib import Path

TASK_IDS = [
    "1db09e4b", "1e51d0b6", "1e81d6f9", "2037f2c7", "20818e16",
    "2072aba6", "22a4bbc2", "22eb0ac0", "23581191", "239be575",
    "253bf280", "264363fd", "27a77e38", "2a5f8217", "2b01abd0",
    "2bee17df", "2c0b0aff", "2cb3571e", "2f0c5170", "310f3251",
]

SEARCH_DIRS = [
    os.path.expanduser("~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation"),
    os.path.expanduser("~/ARC_AMD_TRANSFER/data/ARC-AGI/data/training"),
    os.path.expanduser("~/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation"),
    os.path.expanduser("~/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/training"),
]

COLOR_MAP = {
    0: "#000000",  # black
    1: "#0074D9",  # blue
    2: "#FF4136",  # red
    3: "#2ECC40",  # green
    4: "#FFDC00",  # yellow
    5: "#AAAAAA",  # gray
    6: "#F012BE",  # magenta
    7: "#FF851B",  # orange
    8: "#7FDBFF",  # cyan
    9: "#85144b",  # maroon
}

OUT_DIR = os.path.expanduser("~/arc-puzzle-catalog/viz")
os.makedirs(OUT_DIR, exist_ok=True)


def find_task_file(task_id: str) -> str | None:
    for d in SEARCH_DIRS:
        p = os.path.join(d, f"{task_id}.json")
        if os.path.isfile(p):
            return p
    return None


def grid_to_html(grid: list[list[int]], label: str) -> str:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    cells = ""
    for row in grid:
        cells += "<tr>"
        for val in row:
            color = COLOR_MAP.get(val, "#000000")
            cells += f'<td style="width:20px;height:20px;background:{color};border:1px solid #444;padding:0;"></td>'
        cells += "</tr>"
    return f"""<div style="display:inline-block;vertical-align:top;margin:0 18px 18px 0;">
  <div style="font-weight:bold;margin-bottom:4px;font-size:14px;">{label}</div>
  <table style="border-collapse:collapse;">{cells}</table>
  <div style="font-size:11px;color:#888;margin-top:2px;">{rows}×{cols}</div>
</div>"""


def generate_html(task_id: str, data: dict) -> str:
    train = data.get("train", [])
    pairs_html = ""
    for i, pair in enumerate(train):
        inp = grid_to_html(pair["input"], "Input")
        out = grid_to_html(pair["output"], "Output")
        pairs_html += f"""<div style="margin-bottom:24px;">
  <h3 style="margin:0 0 8px 0;font-family:monospace;">Train {i}</h3>
  <div style="display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap;">
    {inp}{out}
  </div>
</div>"""

    # Also show test inputs (no output)
    test = data.get("test", [])
    for i, pair in enumerate(test):
        inp = grid_to_html(pair["input"], "Input")
        out_html = ""
        if "output" in pair:
            out_html = grid_to_html(pair["output"], "Output")
        else:
            out_html = '<div style="display:inline-block;vertical-align:top;margin:0 18px;"><div style="font-weight:bold;margin-bottom:4px;font-size:14px;">Output</div><div style="color:#888;font-style:italic;">? (to solve)</div></div>'
        pairs_html += f"""<div style="margin-bottom:24px;">
  <h3 style="margin:0 0 8px 0;font-family:monospace;color:#D35400;">Test {i}</h3>
  <div style="display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap;">
    {inp}{out_html}
  </div>
</div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ARC Task {task_id}</title></head>
<body style="background:#1a1a1a;color:#eee;font-family:sans-serif;padding:24px;">
<h1 style="font-family:monospace;">ARC Task: {task_id}</h1>
<p style="color:#888;">{len(train)} training pair(s), {len(test)} test input(s)</p>
<div style="margin-bottom:12px;font-size:12px;color:#aaa;">
  Color key: <span style="color:#000;background:#000;padding:0 6px;border:1px solid #444;">0</span>
  <span style="color:#fff;background:#0074D9;padding:0 6px;">1</span>
  <span style="color:#fff;background:#FF4136;padding:0 6px;">2</span>
  <span style="color:#fff;background:#2ECC40;padding:0 6px;">3</span>
  <span style="color:#000;background:#FFDC00;padding:0 6px;">4</span>
  <span style="color:#fff;background:#AAAAAA;padding:0 6px;">5</span>
  <span style="color:#fff;background:#F012BE;padding:0 6px;">6</span>
  <span style="color:#fff;background:#FF851B;padding:0 6px;">7</span>
  <span style="color:#000;background:#7FDBFF;padding:0 6px;">8</span>
  <span style="color:#fff;background:#85144b;padding:0 6px;">9</span>
</div>
<hr style="border-color:#333;">
{pairs_html}
</body></html>"""


def main():
    generated = 0
    skipped = []
    for task_id in TASK_IDS:
        path = find_task_file(task_id)
        if not path:
            skipped.append(task_id)
            print(f"  SKIP {task_id} — file not found")
            continue
        with open(path) as f:
            data = json.load(f)
        html = generate_html(task_id, data)
        out_path = os.path.join(OUT_DIR, f"{task_id}.html")
        with open(out_path, "w") as f:
            f.write(html)
        n_train = len(data.get("train", []))
        n_test = len(data.get("test", []))
        print(f"  ✓ {task_id}.html  ({n_train} train, {n_test} test)")
        generated += 1

    print(f"\nDone: {generated} generated, {len(skipped)} skipped")
    if skipped:
        print(f"Skipped (not found locally): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
