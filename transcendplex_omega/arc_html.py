"""
ARC HTML Reasoning Generator
Produces a rich dark-theme HTML page for any ARC task, showing:
  - Discovered rule description
  - All training pairs (input → output grids with diff highlights)
  - Test input → predicted output (colour-coded PASS/FAIL banner)
Usage:
    from arc_html import generate_html
    html = generate_html(task_id, task_data, rule_description, predicted_output)
    with open(f"solves/{task_id}/reasoning.html", "w") as f:
        f.write(html)
"""
from typing import List, Optional

Grid = List[List[int]]

COLORS = {
    0:  ("#111111", "#ffffff"),  # black
    1:  ("#0074D9", "#000000"),  # blue
    2:  ("#FF4136", "#000000"),  # red
    3:  ("#2ECC40", "#000000"),  # green
    4:  ("#FFDC00", "#000000"),  # yellow
    5:  ("#AAAAAA", "#000000"),  # grey
    6:  ("#F012BE", "#ffffff"),  # magenta
    7:  ("#FF851B", "#000000"),  # orange
    8:  ("#7FDBFF", "#000000"),  # light blue
    9:  ("#870C25", "#ffffff"),  # maroon
}

COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "grey", 6: "magenta", 7: "orange", 8: "cyan", 9: "maroon",
}


def _cell(v: int, highlight: bool = False, cell_size: int = 24) -> str:
    bg, fg = COLORS.get(v, ("#333", "#fff"))
    border = "3px solid #FFD700" if highlight else "1px solid #444"
    return (
        f'<td style="width:{cell_size}px;height:{cell_size}px;background:{bg};'
        f'border:{border};text-align:center;font-size:11px;color:{fg}">{v}</td>'
    )


def _grid_html(grid: Grid, diff_grid: Optional[Grid] = None,
               cell_size: int = 24) -> str:
    rows = []
    for r, row in enumerate(grid):
        cells = []
        for c, v in enumerate(row):
            highlight = (
                diff_grid is not None
                and r < len(diff_grid)
                and c < len(diff_grid[r])
                and diff_grid[r][c] != grid[r][c]
            )
            cells.append(_cell(v, highlight, cell_size))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return '<table style="border-collapse:collapse;margin:4px">' + "".join(rows) + "</table>"


def _pair_section(label: str, inp: Grid, out: Grid,
                  predicted: Optional[Grid] = None, cell_size: int = 24) -> str:

    parts = [
        f'<div style="display:inline-block;margin:8px;vertical-align:top">'
        f'<div style="font-weight:bold;margin-bottom:4px;color:#aaa">Input</div>'
        + _grid_html(inp, cell_size=cell_size) + "</div>",
        f'<div style="display:inline-block;margin:8px;vertical-align:top;font-size:28px;padding-top:30px">→</div>',
        f'<div style="display:inline-block;margin:8px;vertical-align:top">'
        f'<div style="font-weight:bold;margin-bottom:4px;color:#aaa">Output</div>'
        + _grid_html(out, cell_size=cell_size) + "</div>",
    ]

    if predicted is not None:
        ok = predicted == out
        banner_color = "#2ECC40" if ok else "#FF4136"
        status = "✅ PASS" if ok else "❌ FAIL"
        parts.append(
            f'<div style="display:inline-block;margin:8px;vertical-align:top">'
            f'<div style="font-weight:bold;margin-bottom:4px;color:{banner_color}">'
            f'Predicted {status}</div>'
            + _grid_html(predicted, diff_grid=out, cell_size=cell_size) + "</div>"
        )

    return (
        f'<h3 style="color:#ccc;margin-top:24px">{label}</h3>'
        f'<div style="display:flex;align-items:flex-start;flex-wrap:wrap">{"".join(parts)}</div>'
    )


def _normalize_predictions(predicted_test: Optional[object], test_count: int) -> list[Optional[Grid]]:
    if predicted_test is None:
        return [None] * test_count
    if (
        isinstance(predicted_test, list)
        and predicted_test
        and isinstance(predicted_test[0], list)
        and predicted_test[0]
        and isinstance(predicted_test[0][0], int)
    ):
        return [predicted_test]
    return list(predicted_test)


def generate_html(task_id: str, task_data: dict, rule: str,
                  predicted_test: Optional[object] = None) -> str:
    body_parts = [
        f'<h1 style="color:#FFD700">ARC-AGI-2 · Task <code>{task_id}</code></h1>',
        f'<div style="background:#2a2a2a;padding:16px;border-left:4px solid #FFD700;'
        f'margin-bottom:24px;border-radius:4px">'
        f'<b style="color:#FFD700">Discovered Rule:</b><br>'
        f'<span style="color:#eee">{rule}</span></div>',
    ]

    for i, ex in enumerate(task_data.get("train", [])):
        body_parts.append(_pair_section(f"Training Example {i+1}", ex["input"], ex["output"]))

    test_examples = task_data.get("test", [])
    predictions = _normalize_predictions(predicted_test, len(test_examples))
    for i, test_ex in enumerate(test_examples):
        test_inp = test_ex["input"]
        test_out = test_ex.get("output")
        predicted = predictions[i] if i < len(predictions) else None
        label = f"Test Example {i + 1}"
        if predicted is not None and test_out is not None:
            body_parts.append(_pair_section(label, test_inp, test_out, predicted=predicted))
        elif predicted is not None:
            body_parts.append(
                f'<h3 style="color:#ccc;margin-top:24px">{label} (Prediction Only)</h3>'
                f'<div style="display:flex;align-items:flex-start;flex-wrap:wrap">'
                f'<div style="display:inline-block;margin:8px;vertical-align:top">'
                f'<div style="font-weight:bold;margin-bottom:4px;color:#aaa">Input</div>'
                + _grid_html(test_inp) +
                f'</div><div style="display:inline-block;margin:8px;vertical-align:top;'
                f'font-size:28px;padding-top:30px">→</div>'
                f'<div style="display:inline-block;margin:8px;vertical-align:top">'
                f'<div style="font-weight:bold;margin-bottom:4px;color:#aaa">Predicted</div>'
                + _grid_html(predicted) + "</div></div>"
            )
        elif test_out is not None:
            body_parts.append(_pair_section(label, test_inp, test_out))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>ARC-AGI-2 {task_id}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #1a1a1a; color: #eee; padding: 24px; }}
  code {{ background: #333; padding: 2px 6px; border-radius: 3px; color: #FFD700; }}
  h1, h2, h3 {{ margin: 0 0 12px; }}
</style>
</head>
<body>
{"".join(body_parts)}
<footer style="margin-top:40px;color:#555;font-size:12px">
  Generated by TranscendPlexity ARC-AGI-2 Solver Pipeline
</footer>
</body></html>"""
    return html
