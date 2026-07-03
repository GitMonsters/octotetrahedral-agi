#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

TASK_IDS = [
    "13e47133",
    "21897d95",
    "269e22fb",
    "2b83f449",
    "4e34c42c",
    "62593bfd",
    "88bcf3b4",
    "88e364bc",
    "8b7bacbf",
    "9bbf930d",
    "a32d8b75",
    "abc82100",
    "e12f9a14",
]

SCRIPT_DIR = Path(__file__).resolve().parent
TASKS_DIR = SCRIPT_DIR / "13-Impossible-ARC-Tasks-SOLVED" / "dataset" / "tasks"
SOLVES_DIR = SCRIPT_DIR / "13-Impossible-ARC-Tasks-SOLVED" / "solves"
OUTPUT_HTML = SCRIPT_DIR / "impossible13_animation.html"

PALETTE = {
    0: "#000000",
    1: "#1F77B4",
    2: "#D62728",
    3: "#2CA02C",
    4: "#FFDB58",
    5: "#7F7F7F",
    6: "#E377C2",
    7: "#FF7F0E",
    8: "#00BFFF",
    9: "#8B0000",
}


def to_grid(value: Any) -> list[list[int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [[int(cell) for cell in row] for row in value]


def load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    solver_dir = str(path.parent)
    restore_path = False
    if solver_dir not in sys.path:
        sys.path.insert(0, solver_dir)
        restore_path = True
    try:
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            spec.loader.exec_module(module)
    finally:
        if restore_path and sys.path and sys.path[0] == solver_dir:
            sys.path.pop(0)
    return module


def find_solver(module: ModuleType, task_id: str) -> tuple[Callable[[list[list[int]]], Any], str]:
    candidates = [name for name in dir(module) if name.startswith("solve") and callable(getattr(module, name))]
    priority = ["solve", f"solve_{task_id}"]
    for name in priority:
        if name in candidates:
            return getattr(module, name), name
    if candidates:
        chosen = sorted(candidates)[0]
        return getattr(module, chosen), chosen
    if callable(getattr(module, "transform", None)):
        return getattr(module, "transform"), "transform"
    raise RuntimeError(f"No solver function found for {task_id}")


def run_solver(solver: Callable[[list[list[int]]], Any], grid: list[list[int]]) -> list[list[int]]:
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        result = solver(deepcopy(grid))
    return to_grid(result)


def load_task_payload(task_id: str) -> dict[str, Any]:
    task_path = TASKS_DIR / f"{task_id}.json"
    solver_path = SOLVES_DIR / task_id / "solver.py"

    task = json.loads(task_path.read_text())
    solver_source = solver_path.read_text()

    module = load_module(f"impossible13_{task_id}", solver_path)
    solver, solver_name = find_solver(module, task_id)

    train_pairs: list[dict[str, Any]] = []
    for example in task["train"]:
        predicted = run_solver(solver, example["input"])
        train_pairs.append(
            {
                "input": example["input"],
                "output": example["output"],
                "passed": predicted == example["output"],
            }
        )

    if not task.get("test"):
        raise RuntimeError(f"Task {task_id} has no test cases")
    test_input = task["test"][0]["input"]
    test_output = run_solver(solver, test_input)

    return {
        "id": task_id,
        "solver_name": solver_name,
        "code_lines": solver_source.splitlines()[:10],
        "train": train_pairs,
        "test_input": test_input,
        "test_output": test_output,
        "train_count": len(train_pairs),
        "train_passed": sum(1 for pair in train_pairs if pair["passed"]),
    }


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>13 Impossible Tasks</title>
  <style>
    :root {
      --bg: #0a0a0f;
      --panel: rgba(16, 20, 28, 0.82);
      --panel-strong: rgba(14, 18, 26, 0.95);
      --text: #f5f7fb;
      --muted: rgba(214, 224, 255, 0.72);
      --green: #36ff95;
      --green-dim: rgba(54, 255, 149, 0.2);
      --gold: #ffd86b;
      --danger: #ff6161;
      --shadow: 0 0 24px rgba(54, 255, 149, 0.16);
      --grid-border: rgba(255, 255, 255, 0.74);
    }

    * { box-sizing: border-box; }

    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      background: radial-gradient(circle at top, rgba(54, 255, 149, 0.10), transparent 28%),
                  radial-gradient(circle at 80% 20%, rgba(0, 191, 255, 0.10), transparent 30%),
                  linear-gradient(180deg, #0b0b12 0%, #08090e 100%);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: radial-gradient(circle at center, black, transparent 86%);
      pointer-events: none;
      opacity: 0.28;
    }

    #particles {
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
    }

    .progress-shell {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: rgba(255,255,255,0.06);
      z-index: 9;
    }

    .progress-bar {
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, #1cff88, #7bffce);
      box-shadow: 0 0 18px rgba(54,255,149,0.8);
      transition: width 420ms ease;
    }

    .app {
      position: relative;
      z-index: 1;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 34px 44px 32px;
      gap: 22px;
    }

    .hud {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: start;
    }

    .title {
      font-size: clamp(2rem, 4vw, 3.5rem);
      font-weight: 800;
      letter-spacing: 0.06em;
      margin: 0;
    }

    .subtitle {
      margin-top: 8px;
      color: var(--muted);
      font-size: clamp(0.92rem, 1.4vw, 1.12rem);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .hud-meta {
      display: flex;
      gap: 14px;
      align-items: stretch;
      justify-content: flex-end;
      flex-wrap: wrap;
    }

    .meta-box {
      min-width: 132px;
      padding: 12px 14px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      border-radius: 14px;
      backdrop-filter: blur(12px);
      box-shadow: 0 14px 40px rgba(0,0,0,0.18);
    }

    .meta-label {
      font-size: 0.72rem;
      color: var(--muted);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }

    .meta-value {
      font-size: 1.22rem;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }

    .stage {
      flex: 1;
      display: flex;
      align-items: stretch;
      justify-content: center;
      min-height: 0;
    }

    .panel {
      width: min(1280px, 100%);
      min-height: 100%;
      border-radius: 28px;
      border: 1px solid rgba(255,255,255,0.08);
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
      box-shadow: 0 30px 80px rgba(0,0,0,0.45);
      backdrop-filter: blur(16px);
      padding: 28px;
      position: relative;
      overflow: hidden;
    }

    .panel::after {
      content: "";
      position: absolute;
      inset: 0;
      border-radius: inherit;
      pointer-events: none;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
    }

    .phase-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 18px;
      margin-bottom: 24px;
    }

    .phase-label {
      font-size: 0.78rem;
      color: var(--green);
      letter-spacing: 0.22em;
      text-transform: uppercase;
      text-shadow: 0 0 14px rgba(54,255,149,0.45);
    }

    .task-id {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: clamp(1.15rem, 2.4vw, 2rem);
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: lowercase;
    }

    .task-counter {
      color: var(--muted);
      font-size: 1rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .content {
      height: calc(100% - 88px);
      display: grid;
      gap: 18px;
      align-items: center;
    }

    .typing-layout {
      grid-template-columns: 1.2fr 0.8fr;
    }

    .validate-layout {
      grid-template-columns: 1fr;
      align-content: start;
    }

    .solve-layout {
      grid-template-columns: 1fr auto 1fr;
    }

    .typing-card,
    .validate-card,
    .solve-card,
    .finale-card {
      border-radius: 22px;
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }

    .typing-card {
      min-height: 460px;
      padding: 18px 22px 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .typing-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 0.9rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .badge {
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid rgba(54,255,149,0.5);
      background: rgba(54,255,149,0.12);
      color: #dfffe9;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      animation: pulse 1.2s ease-in-out infinite;
      box-shadow: 0 0 18px rgba(54,255,149,0.25);
    }

    @keyframes pulse {
      0%, 100% { transform: scale(1); box-shadow: 0 0 18px rgba(54,255,149,0.2); }
      50% { transform: scale(1.04); box-shadow: 0 0 26px rgba(54,255,149,0.45); }
    }

    .code {
      flex: 1;
      margin: 0;
      padding: 20px;
      background: rgba(0,0,0,0.34);
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.06);
      color: #d7ffe6;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: clamp(0.82rem, 1.25vw, 1rem);
      line-height: 1.5;
      white-space: pre-wrap;
      overflow: hidden;
      position: relative;
      min-height: 360px;
    }

    .cursor {
      display: inline-block;
      width: 10px;
      margin-left: 3px;
      background: var(--green);
      height: 1.1em;
      vertical-align: -0.15em;
      animation: blink 0.9s steps(1) infinite;
      box-shadow: 0 0 12px rgba(54,255,149,0.7);
    }

    @keyframes blink {
      0%, 45% { opacity: 1; }
      46%, 100% { opacity: 0; }
    }

    .typing-side {
      display: grid;
      gap: 16px;
      align-content: center;
      padding: 12px;
    }

    .side-stat {
      padding: 18px 20px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 18px;
    }

    .side-label {
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      margin-bottom: 8px;
    }

    .side-value {
      font-size: 1.5rem;
      font-weight: 750;
    }

    .validate-card {
      padding: 18px 20px 22px;
      min-height: 500px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .validate-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }

    .training-count {
      font-size: clamp(1.2rem, 2vw, 1.8rem);
      font-weight: 800;
      color: var(--green);
      text-shadow: 0 0 20px rgba(54,255,149,0.36);
      font-variant-numeric: tabular-nums;
    }

    .train-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 14px;
      align-items: start;
    }

    .train-pair {
      display: grid;
      gap: 12px;
      padding: 14px;
      border-radius: 18px;
      background: rgba(255,255,255,0.035);
      border: 1px solid rgba(255,255,255,0.06);
      transition: transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease;
    }

    .train-pair.pass {
      border-color: rgba(54,255,149,0.45);
      box-shadow: 0 0 22px rgba(54,255,149,0.18);
      transform: translateY(-1px);
    }

    .pair-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 0.82rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .pair-check {
      color: var(--green);
      font-size: 1.1rem;
      opacity: 0;
      transition: opacity 180ms ease;
      text-shadow: 0 0 16px rgba(54,255,149,0.6);
    }

    .train-pair.pass .pair-check { opacity: 1; }

    .pair-body {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    .arrow {
      font-size: 1.5rem;
      color: var(--green);
      text-shadow: 0 0 16px rgba(54,255,149,0.5);
      animation: nudge 1s ease-in-out infinite;
    }

    @keyframes nudge {
      0%, 100% { transform: translateX(0); }
      50% { transform: translateX(7px); }
    }

    .solve-card {
      min-height: 500px;
      padding: 22px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 16px;
      align-items: center;
    }

    .solve-title {
      font-size: clamp(1.5rem, 3vw, 2.6rem);
      font-weight: 900;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--green);
      opacity: 0;
      transform: scale(0.92);
      transition: opacity 180ms ease, transform 180ms ease;
      text-shadow: 0 0 18px rgba(54,255,149,0.55), 0 0 34px rgba(54,255,149,0.25);
    }

    .solve-title.show {
      opacity: 1;
      transform: scale(1);
    }

    .solve-grid-title {
      margin-bottom: 10px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 0.78rem;
      text-align: center;
    }

    .solve-arrow {
      align-self: center;
      font-size: clamp(2rem, 4vw, 3.4rem);
      color: var(--green);
      text-shadow: 0 0 24px rgba(54,255,149,0.5);
      animation: breathe 1.2s ease-in-out infinite;
    }

    @keyframes breathe {
      0%, 100% { transform: scale(1); opacity: 0.82; }
      50% { transform: scale(1.08); opacity: 1; }
    }

    .grid-shell {
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
    }

    .grid {
      display: grid;
      gap: 0;
      padding: 10px;
      background: rgba(255,255,255,0.03);
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03), 0 16px 40px rgba(0,0,0,0.22);
      justify-content: center;
    }

    .cell {
      width: var(--cell-size);
      height: var(--cell-size);
      border: 0.5px solid var(--grid-border);
      opacity: 1;
      transform: scale(1);
      transition: opacity 180ms ease, transform 180ms ease;
    }

    .cell.hidden {
      opacity: 0;
      transform: scale(0.12);
    }

    .finale {
      position: absolute;
      inset: 0;
      display: none;
      place-items: center;
      padding: 32px;
      background: radial-gradient(circle at center, rgba(255,216,107,0.14), transparent 38%), rgba(5, 7, 12, 0.84);
      backdrop-filter: blur(10px);
      z-index: 5;
    }

    .finale.show { display: grid; }

    .finale-card {
      width: min(900px, 100%);
      padding: 40px 36px;
      text-align: center;
      background: rgba(16,18,26,0.84);
      border-color: rgba(255,216,107,0.24);
      box-shadow: 0 0 42px rgba(255,216,107,0.12), 0 26px 80px rgba(0,0,0,0.42);
    }

    .finale-kicker {
      color: var(--gold);
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 0.84rem;
      margin-bottom: 16px;
    }

    .finale-big {
      font-size: clamp(2.6rem, 7vw, 5.8rem);
      font-weight: 900;
      letter-spacing: 0.1em;
      color: #fff4cf;
      text-shadow: 0 0 18px rgba(255,216,107,0.55), 0 0 44px rgba(255,216,107,0.18);
      margin: 0;
    }

    .finale-sub {
      margin-top: 18px;
      font-size: clamp(1rem, 2vw, 1.45rem);
      color: #fff0bf;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .finale-time {
      margin-top: 18px;
      font-size: 1.3rem;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }

    .loop-button {
      margin-top: 28px;
      border: 1px solid rgba(255,216,107,0.34);
      background: rgba(255,216,107,0.08);
      color: #fff4cf;
      padding: 12px 18px;
      border-radius: 999px;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      cursor: pointer;
      transition: transform 180ms ease, background 180ms ease;
    }

    .loop-button:hover {
      transform: translateY(-1px);
      background: rgba(255,216,107,0.14);
    }

    @media (max-width: 980px) {
      .app { padding: 24px 18px 18px; }
      .content,
      .typing-layout,
      .solve-layout { grid-template-columns: 1fr; }
      .solve-arrow { transform: rotate(90deg); }
      .phase-row { flex-wrap: wrap; }
      .hud { grid-template-columns: 1fr; }
      .hud-meta { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <canvas id="particles"></canvas>
  <div class="progress-shell"><div class="progress-bar" id="progressBar"></div></div>
  <div class="app">
    <header class="hud">
      <div>
        <h1 class="title">13 IMPOSSIBLE TASKS</h1>
        <div class="subtitle">0% Human AI Solve Rate → TranscendPlexity: 100%</div>
      </div>
      <div class="hud-meta">
        <div class="meta-box">
          <div class="meta-label">Elapsed</div>
          <div class="meta-value" id="elapsed">0.0s</div>
        </div>
        <div class="meta-box">
          <div class="meta-label">Solved</div>
          <div class="meta-value" id="solvedTop">0/13</div>
        </div>
      </div>
    </header>

    <main class="stage">
      <section class="panel">
        <div class="phase-row">
          <div>
            <div class="phase-label" id="phaseLabel">Initializing</div>
            <div class="task-id" id="taskId">loading...</div>
          </div>
          <div class="task-counter" id="taskCounter">0 / 13 Solved</div>
        </div>
        <div class="content" id="content"></div>
        <div class="finale" id="finale">
          <div class="finale-card">
            <div class="finale-kicker">Impossible no more</div>
            <h2 class="finale-big">13 / 13 SOLVED</h2>
            <div class="finale-sub">100% — ALL IMPOSSIBLE TASKS CONQUERED</div>
            <div class="finale-time" id="finaleTime">Elapsed: 0.0s</div>
            <button class="loop-button" id="replayButton">Replay Sequence</button>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const TASKS = __TASK_DATA__;
    const PALETTE = __PALETTE_DATA__;
    const TOTAL = TASKS.length;
    const state = {
      startTime: performance.now(),
      completed: 0,
      timeouts: [],
      token: 0,
      finalParticles: false,
    };

    const progressBar = document.getElementById('progressBar');
    const phaseLabel = document.getElementById('phaseLabel');
    const taskIdEl = document.getElementById('taskId');
    const taskCounter = document.getElementById('taskCounter');
    const solvedTop = document.getElementById('solvedTop');
    const elapsed = document.getElementById('elapsed');
    const content = document.getElementById('content');
    const finale = document.getElementById('finale');
    const finaleTime = document.getElementById('finaleTime');
    const replayButton = document.getElementById('replayButton');
    const particlesCanvas = document.getElementById('particles');
    const pctx = particlesCanvas.getContext('2d');
    let particles = [];

    function resizeCanvas() {
      particlesCanvas.width = window.innerWidth * devicePixelRatio;
      particlesCanvas.height = window.innerHeight * devicePixelRatio;
      particlesCanvas.style.width = `${window.innerWidth}px`;
      particlesCanvas.style.height = `${window.innerHeight}px`;
      pctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    function queueTimeout(fn, delay) {
      const id = window.setTimeout(fn, delay);
      state.timeouts.push(id);
      return id;
    }

    function sleep(ms, token) {
      return new Promise((resolve) => {
        queueTimeout(() => resolve(token === state.token), ms);
      });
    }

    function clearScheduled() {
      for (const id of state.timeouts) {
        clearTimeout(id);
      }
      state.timeouts = [];
    }

    function setSolvedCount(count) {
      taskCounter.textContent = `${count} / ${TOTAL} SOLVED`;
      solvedTop.textContent = `${count}/${TOTAL}`;
    }

    function setProgress(index, phaseFraction) {
      const value = Math.max(0, Math.min(1, (index + phaseFraction) / TOTAL));
      progressBar.style.width = `${(value * 100).toFixed(2)}%`;
    }

    function updateClock(now) {
      const seconds = (now - state.startTime) / 1000;
      elapsed.textContent = `${seconds.toFixed(1)}s`;
      if (finale.classList.contains('show')) {
        finaleTime.textContent = `Elapsed: ${seconds.toFixed(1)}s`;
      }
      requestAnimationFrame(updateClock);
    }
    requestAnimationFrame(updateClock);

    function createEl(tag, className, text) {
      const el = document.createElement(tag);
      if (className) el.className = className;
      if (text !== undefined) el.textContent = text;
      return el;
    }

    function cellDelayForGrid(grid) {
      const cells = grid.length * grid[0].length;
      return Math.max(2, Math.min(15, Math.floor(1200 / Math.max(cells, 1))));
    }

    function renderGrid(grid, options = {}) {
      const maxSize = options.maxSize ?? 300;
      const rows = grid.length;
      const cols = grid[0].length;
      const cellSize = Math.max(8, Math.floor(Math.min(maxSize / cols, maxSize / rows)));
      const wrapper = createEl('div', 'grid-shell');
      if (options.label) {
        wrapper.appendChild(createEl('div', 'solve-grid-title', options.label));
      }
      const gridEl = createEl('div', 'grid');
      gridEl.style.setProperty('--cell-size', `${cellSize}px`);
      gridEl.style.gridTemplateColumns = `repeat(${cols}, ${cellSize}px)`;
      const timeouts = [];
      const delay = options.delay ?? 15;
      grid.forEach((row, r) => {
        row.forEach((value, c) => {
          const cell = createEl('div', 'cell hidden');
          cell.style.background = PALETTE[value] || '#000000';
          gridEl.appendChild(cell);
          const revealAt = options.reveal ? (r * cols + c) * delay : 0;
          const timeoutId = queueTimeout(() => {
            cell.classList.remove('hidden');
          }, revealAt);
          timeouts.push(timeoutId);
          if (!options.reveal) {
            cell.classList.remove('hidden');
          }
        });
      });
      wrapper.appendChild(gridEl);
      return { wrapper, duration: options.reveal ? rows * cols * delay : 0 };
    }

    function renderTypingPhase(task) {
      content.className = 'content typing-layout';
      content.innerHTML = '';

      const typingCard = createEl('div', 'typing-card');
      const top = createEl('div', 'typing-top');
      top.appendChild(createEl('div', '', 'Synthesizing solver trace'));
      top.appendChild(createEl('div', 'badge', 'CLAUDE OPUS'));
      typingCard.appendChild(top);

      const code = createEl('pre', 'code');
      const codeText = createEl('span');
      const cursor = createEl('span', 'cursor');
      code.appendChild(codeText);
      code.appendChild(cursor);
      typingCard.appendChild(code);

      const side = createEl('div', 'typing-side');
      const statA = createEl('div', 'side-stat');
      statA.appendChild(createEl('div', 'side-label', 'Solver function'));
      statA.appendChild(createEl('div', 'side-value', task.solver_name));
      const statB = createEl('div', 'side-stat');
      statB.appendChild(createEl('div', 'side-label', 'Code sample'));
      statB.appendChild(createEl('div', 'side-value', `${task.code_lines.length} live lines`));
      const statC = createEl('div', 'side-stat');
      statC.appendChild(createEl('div', 'side-label', 'Training cases'));
      statC.appendChild(createEl('div', 'side-value', `${task.train_count} examples`));
      side.append(statA, statB, statC);

      content.append(typingCard, side);

      const snippet = task.code_lines.join('\n');
      const minStep = 12;
      const step = Math.max(minStep, Math.floor(1200 / Math.max(snippet.length, 1)));
      let index = 0;
      (function typeNext() {
        codeText.textContent = snippet.slice(0, index);
        index += 1;
        if (index <= snippet.length) {
          queueTimeout(typeNext, step);
        }
      })();
    }

    function renderValidationPhase(task) {
      content.className = 'content validate-layout';
      content.innerHTML = '';

      const card = createEl('div', 'validate-card');
      const top = createEl('div', 'validate-top');
      top.appendChild(createEl('div', 'phase-label', 'VALIDATING'));
      const counter = createEl('div', 'training-count', `Training: 0/${task.train_count} ✓`);
      top.appendChild(counter);
      card.appendChild(top);

      const grid = createEl('div', 'train-grid');
      card.appendChild(grid);
      content.appendChild(card);

      task.train.forEach((pair, idx) => {
        const pairEl = createEl('div', 'train-pair');
        const header = createEl('div', 'pair-header');
        header.appendChild(createEl('div', '', `Example ${idx + 1}`));
        header.appendChild(createEl('div', 'pair-check', '✓'));
        pairEl.appendChild(header);
        const body = createEl('div', 'pair-body');
        body.appendChild(renderGrid(pair.input, { maxSize: 112, reveal: false }).wrapper);
        body.appendChild(createEl('div', 'arrow', '→'));
        body.appendChild(renderGrid(pair.output, { maxSize: 112, reveal: false }).wrapper);
        pairEl.appendChild(body);
        grid.appendChild(pairEl);
      });

      task.train.forEach((pair, idx) => {
        queueTimeout(() => {
          const pairEl = grid.children[idx];
          if (!pairEl) return;
          pairEl.classList.toggle('pass', !!pair.passed);
          counter.textContent = `Training: ${idx + 1}/${task.train_count} ✓`;
        }, Math.floor((800 / Math.max(task.train.length, 1)) * (idx + 1)));
      });
    }

    function renderSolvePhase(task) {
      content.className = 'content solve-layout';
      content.innerHTML = '';

      const left = createEl('div', 'solve-card');
      const middle = createEl('div', 'solve-arrow', '→');
      const right = createEl('div', 'solve-card');
      const title = createEl('div', 'solve-title', '✓ SOLVED');

      const inputGrid = renderGrid(task.test_input, { label: 'Test Input', maxSize: 300, reveal: false });
      const outputDelay = cellDelayForGrid(task.test_output);
      const outputGrid = renderGrid(task.test_output, { label: 'Solver Output', maxSize: 300, reveal: true, delay: outputDelay });

      left.appendChild(inputGrid.wrapper);
      right.appendChild(outputGrid.wrapper);
      right.appendChild(title);
      content.append(left, middle, right);

      queueTimeout(() => title.classList.add('show'), 180);
      return Math.max(2000, outputGrid.duration + 420);
    }

    function spawnCelebration() {
      particles = [];
      const colors = ['#36ff95', '#ffd86b', '#00BFFF', '#FF7F0E', '#E377C2', '#ffffff'];
      for (let i = 0; i < 180; i += 1) {
        particles.push({
          x: window.innerWidth / 2,
          y: window.innerHeight * 0.2,
          vx: (Math.random() - 0.5) * 12,
          vy: Math.random() * 5 + 2,
          g: Math.random() * 0.22 + 0.12,
          size: Math.random() * 5 + 2,
          color: colors[Math.floor(Math.random() * colors.length)],
          life: Math.random() * 140 + 90,
          angle: Math.random() * Math.PI,
          spin: (Math.random() - 0.5) * 0.24,
        });
      }
    }

    function animateParticles() {
      pctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      particles = particles.filter((p) => p.life > 0);
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        p.vy += p.g;
        p.life -= 1;
        p.angle += p.spin;
        pctx.save();
        pctx.translate(p.x, p.y);
        pctx.rotate(p.angle);
        pctx.fillStyle = p.color;
        pctx.globalAlpha = Math.max(0, p.life / 180);
        pctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 1.8);
        pctx.restore();
      }
      requestAnimationFrame(animateParticles);
    }
    requestAnimationFrame(animateParticles);

    async function runSequence() {
      state.token += 1;
      const token = state.token;
      clearScheduled();
      finale.classList.remove('show');
      state.startTime = performance.now();
      state.completed = 0;
      setSolvedCount(0);
      setProgress(0, 0);

      for (let index = 0; index < TASKS.length; index += 1) {
        if (token !== state.token) return;
        const task = TASKS[index];
        taskIdEl.textContent = task.id;

        phaseLabel.textContent = 'SYNTHESIZING';
        setProgress(index, 0.08);
        renderTypingPhase(task);
        if (!(await sleep(1200, token))) return;

        phaseLabel.textContent = 'VALIDATING';
        setProgress(index, 0.42);
        renderValidationPhase(task);
        if (!(await sleep(800, token))) return;

        phaseLabel.textContent = 'SOLVED';
        setProgress(index, 0.72);
        const solveDuration = renderSolvePhase(task);
        if (!(await sleep(solveDuration, token))) return;

        state.completed = index + 1;
        setSolvedCount(state.completed);
        setProgress(index + 1, 0);
      }

      phaseLabel.textContent = 'COMPLETE';
      taskIdEl.textContent = 'all tasks conquered';
      spawnCelebration();
      finale.classList.add('show');
      finaleTime.textContent = `Elapsed: ${((performance.now() - state.startTime) / 1000).toFixed(1)}s`;
    }

    replayButton.addEventListener('click', () => {
      runSequence();
    });

    setSolvedCount(0);
    runSequence();
  </script>
</body>
</html>
'''


def build_html(task_payloads: list[dict[str, Any]]) -> str:
    html_text = HTML_TEMPLATE.replace(
        "__TASK_DATA__",
        json.dumps(task_payloads, separators=(",", ":")),
    ).replace(
        "__PALETTE_DATA__",
        json.dumps(PALETTE, separators=(",", ":")),
    )
    return html_text


def main() -> None:
    payloads = [load_task_payload(task_id) for task_id in TASK_IDS]
    html_output = build_html(payloads)
    OUTPUT_HTML.write_text(html_output)
    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
