from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from arc_utils import TASK_ID, TASK_PATH, load_task
from arc_solver import generate_test_predictions, validate_training_examples

COLOR_HEX = {
    0: '#000000',
    1: '#0074D9',
    2: '#FF4136',
    3: '#2ECC40',
    4: '#FFDC00',
    5: '#AAAAAA',
    6: '#F012BE',
    7: '#FF851B',
    8: '#7FDBFF',
    9: '#870C25',
}


def render_grid(grid: list[list[int]], pixel_size: int = 18) -> str:
    rows = []
    for row in grid:
        cells = ''.join(
            f"<span title='{value}' style='display:inline-block;width:{pixel_size}px;height:{pixel_size}px;background:{COLOR_HEX.get(value, '#111')};border:1px solid #222'></span>"
            for value in row
        )
        rows.append(f"<div style='line-height:0'>{cells}</div>")
    return ''.join(rows)


def save_task_report(task_path: str | Path = TASK_PATH, output_path: str | Path | None = None) -> Path:
    task = load_task(task_path)
    train_results = validate_training_examples(task)
    test_results = generate_test_predictions(task)
    sections = [
        '<html><body style="background:#111;color:#eee;font-family:sans-serif">',
        f'<h1>ARC task {html.escape(TASK_ID)}</h1>',
    ]
    for index, example in enumerate(task['train']):
        sections.append(f'<h2>Train {index}</h2><div style="display:flex;gap:24px">')
        sections.append(f"<div><h3>Input</h3>{render_grid(example['input'])}</div>")
        sections.append(f"<div><h3>Expected</h3>{render_grid(example['output'])}</div>")
        sections.append(f"<div><h3>Prediction</h3>{render_grid(train_results[index]['prediction'])}</div>")
        sections.append('</div>')
    for index, example in enumerate(task['test']):
        sections.append(f'<h2>Test {index}</h2><div style="display:flex;gap:24px">')
        sections.append(f"<div><h3>Input</h3>{render_grid(example['input'])}</div>")
        sections.append(f"<div><h3>Prediction</h3>{render_grid(test_results[index]['prediction'])}</div>")
        if 'output' in example:
            sections.append(f"<div><h3>Reference</h3>{render_grid(example['output'])}</div>")
        sections.append('</div>')
    sections.append('</body></html>')
    destination = Path(output_path) if output_path else CURRENT_DIR / f'{TASK_ID}_report.html'
    destination.write_text(''.join(sections))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f'Render ARC task {TASK_ID}')
    parser.add_argument('--task', default=str(TASK_PATH), help='Path to the task JSON file')
    parser.add_argument('--output', help='HTML output path')
    args = parser.parse_args(argv)
    output_path = save_task_report(args.task, args.output)
    print(output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
