"""Benchmark report generator: HTML (Chart.js), CSV, Markdown, and live dashboard.

Outputs written to benchmark_results/ by default:
  report.html  — interactive dark-theme dashboard with six Chart.js bar charts
  results.csv  — flat CSV for spreadsheet / pandas analysis
  summary.md   — Markdown executive summary with key-finding bullets
  results.json — raw JSON results (written by benchmark_suite.py)

Optional live dashboard (requires fastapi + uvicorn):
  GET /dashboard   — auto-refreshing HTML page
  GET /api/results — latest results as JSON
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("benchmark_results")

# ---------------------------------------------------------------------------
# HTML report (Chart.js via CDN, no extra Python deps)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OctoTetrahedral AGI \u2014 LLM Benchmark Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
          background:#0d1117;color:#e6edf3;padding:24px}}
    h1{{color:#58a6ff;margin-bottom:4px}}
    h2{{color:#79c0ff;border-bottom:1px solid #30363d;padding-bottom:6px;
        margin:24px 0 12px}}
    .meta{{color:#8b949e;font-size:.85em;margin-bottom:24px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}}
    .card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
    .chart-wrap{{position:relative;height:260px}}
    table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:.85em}}
    th{{background:#21262d;color:#8b949e;text-align:left;padding:8px 12px;
        white-space:nowrap}}
    td{{padding:8px 12px;border-bottom:1px solid #21262d;white-space:nowrap}}
    tr:hover td{{background:#1c2128}}
    .good{{color:#3fb950}} .ok{{color:#d29922}} .bad{{color:#f85149}}
    .hl{{color:#58a6ff;font-weight:600}}
    a{{color:#58a6ff}}
  </style>
</head>
<body>
  <h1>\U0001f680 OctoTetrahedral AGI \u2014 LLM Benchmark Report</h1>
  <p class="meta">Generated: {timestamp}&nbsp;\u00b7&nbsp;{n_models} models&nbsp;\u00b7&nbsp;6 scenarios</p>

  <h2>\U0001f4ca Summary</h2>
  {summary_table_html}

  <h2>\U0001f4c8 Charts</h2>
  <div class="grid">
    <div class="card"><h2>\u23f1 Latency (ms, lower \u2193)</h2>
      <div class="chart-wrap"><canvas id="cLatency"></canvas></div></div>
    <div class="card"><h2>\u26a1 Throughput (req/s, higher \u2191)</h2>
      <div class="chart-wrap"><canvas id="cThroughput"></canvas></div></div>
    <div class="card"><h2>\U0001f3af Accuracy (higher \u2191)</h2>
      <div class="chart-wrap"><canvas id="cAccuracy"></canvas></div></div>
    <div class="card"><h2>\U0001f4b0 Cost / 1M tokens USD (lower \u2193)</h2>
      <div class="chart-wrap"><canvas id="cCost"></canvas></div></div>
    <div class="card"><h2>\U0001f50b Energy Wh/1K tokens (lower \u2193)</h2>
      <div class="chart-wrap"><canvas id="cEnergy"></canvas></div></div>
    <div class="card"><h2>\U0001f9e0 Memory MB (lower \u2193)</h2>
      <div class="chart-wrap"><canvas id="cMemory"></canvas></div></div>
  </div>

  <h2>\U0001f4cb Detailed Scenario Results</h2>
  {detail_table_html}

  <script>
  const LABELS = {labels};
  const COLORS = {colors};
  const DEFAULTS = {{
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{label:c=>c.parsed.y.toFixed(3)+" "+c.dataset.unit}}}}}},
    scales:{{
      x:{{ticks:{{color:"#8b949e",maxRotation:30}}}},
      y:{{ticks:{{color:"#8b949e"}}}}
    }}
  }};
  function bar(id, data, label, unit) {{
    new Chart(document.getElementById(id), {{
      type:"bar",
      data:{{labels:LABELS,datasets:[{{label,data,backgroundColor:COLORS,
              borderRadius:4,unit}}]}},
      options:DEFAULTS
    }});
  }}
  bar("cLatency",    {latencies},   "Latency ms",      "ms");
  bar("cThroughput", {throughputs}, "Req/s",            "rps");
  bar("cAccuracy",   {accuracies},  "Accuracy",         "");
  bar("cCost",       {costs},       "Cost USD/1M",      "$");
  bar("cEnergy",     {energies},    "Wh per 1K tokens", "Wh");
  bar("cMemory",     {memories},    "Memory MB",        "MB");
  </script>
</body>
</html>"""


def _badge_class(value: float, good: float, bad: float, lower_is_better: bool) -> str:
    if lower_is_better:
        return "good" if value <= good else ("bad" if value >= bad else "ok")
    return "good" if value >= good else ("bad" if value <= bad else "ok")


def _html_summary_table(metrics_by_model: dict[str, dict[str, Any]]) -> str:
    headers = [
        "Model", "Latency ms", "p99 ms", "Throughput rps",
        "Tokens/s", "Accuracy", "Cost/1M $", "Memory MB", "Energy Wh/1K",
    ]
    rows = ["<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"]
    for model, m in metrics_by_model.items():
        is_octo = "octotetrahedral" in model
        mc = ' class="hl"' if is_octo else ""
        lat = m.get("latency_ms", 0.0)
        lat_c = _badge_class(lat, 300, 2000, True)
        acc = m.get("accuracy", 0.0)
        acc_c = _badge_class(acc, 0.7, 0.4, False)
        rows.append(
            f"<tr>"
            f'<td{mc}>{model}</td>'
            f'<td class="{lat_c}">{lat:.1f}</td>'
            f'<td>{m.get("latency_p99_ms", 0.0):.1f}</td>'
            f'<td>{m.get("throughput_rps", 0.0):.2f}</td>'
            f'<td>{m.get("tokens_per_sec", 0.0):.1f}</td>'
            f'<td class="{acc_c}">{acc:.3f}</td>'
            f'<td>{m.get("cost_per_1m_tokens_usd", 0.0):.4f}</td>'
            f'<td>{m.get("memory_mb", 0.0):.1f}</td>'
            f'<td>{m.get("energy_wh_per_1k_tokens", 0.0):.4f}</td>'
            f"</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _html_detail_table(metrics_by_model: dict[str, dict[str, Any]]) -> str:
    rows = [
        "<table><thead><tr>"
        "<th>Model</th><th>Scenario</th><th>Metric</th><th>Value</th>"
        "</tr></thead><tbody>"
    ]
    for model, m in metrics_by_model.items():
        for sc_name, sc_data in m.get("scenarios", {}).items():
            if not isinstance(sc_data, dict) or "error" in sc_data:
                continue
            for key, val in sc_data.items():
                if key == "scenario":
                    continue
                if isinstance(val, (int, float)):
                    rows.append(
                        f"<tr><td>{model}</td><td>{sc_name}</td>"
                        f"<td>{key}</td><td>{val:.4f}</td></tr>"
                    )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def generate_html_report(
    metrics_by_model: dict[str, dict[str, Any]],
    output_path: Path | str = RESULTS_DIR / "report.html",
) -> str:
    """Generate an HTML report with Chart.js charts; return the output path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = list(metrics_by_model.keys())
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    palette = [
        "#58a6ff", "#3fb950", "#d29922", "#f85149",
        "#bc8cff", "#79c0ff", "#ffa657", "#ff7b72",
    ]
    colors = json.dumps([palette[i % len(palette)] for i in range(len(models))])

    html = _HTML_TEMPLATE.format(
        timestamp=ts,
        n_models=len(models),
        summary_table_html=_html_summary_table(metrics_by_model),
        detail_table_html=_html_detail_table(metrics_by_model),
        labels=json.dumps(models),
        colors=colors,
        latencies=json.dumps([metrics_by_model[m].get("latency_ms", 0.0) for m in models]),
        throughputs=json.dumps([metrics_by_model[m].get("throughput_rps", 0.0) for m in models]),
        accuracies=json.dumps([metrics_by_model[m].get("accuracy", 0.0) for m in models]),
        costs=json.dumps([metrics_by_model[m].get("cost_per_1m_tokens_usd", 0.0) for m in models]),
        energies=json.dumps([metrics_by_model[m].get("energy_wh_per_1k_tokens", 0.0) for m in models]),
        memories=json.dumps([metrics_by_model[m].get("memory_mb", 0.0) for m in models]),
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report \u2192 %s", output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def generate_markdown_summary(
    metrics_by_model: dict[str, dict[str, Any]],
    output_path: Path | str = RESULTS_DIR / "summary.md",
) -> str:
    """Generate a Markdown executive summary; return the output path."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    models = list(metrics_by_model.keys())

    lines = [
        "# OctoTetrahedral AGI \u2014 LLM Benchmark Summary",
        f"_Generated: {ts}_",
        "",
        "## Overview",
        "",
        f"Production benchmark comparing **{len(models)} models** across "
        "6 scenarios: single-inference latency, batch processing, "
        "concurrent requests, long-context handling, few-shot learning, "
        "and reasoning (MMLU + ARC).",
        "",
        "## Results",
        "",
        "| Model | Latency ms | p99 ms | Throughput rps | Tokens/s | "
        "Accuracy | Cost/1M $ | Memory MB |",
        "|-------|-----------|--------|----------------|----------|"
        "---------|-----------|-----------|",
    ]

    for model, m in metrics_by_model.items():
        lines.append(
            f"| {model} "
            f"| {m.get('latency_ms', 0):.1f} "
            f"| {m.get('latency_p99_ms', 0):.1f} "
            f"| {m.get('throughput_rps', 0):.2f} "
            f"| {m.get('tokens_per_sec', 0):.1f} "
            f"| {m.get('accuracy', 0):.3f} "
            f"| {m.get('cost_per_1m_tokens_usd', 0):.4f} "
            f"| {m.get('memory_mb', 0):.1f} |"
        )

    lines += ["", "## Key Findings", ""]

    if metrics_by_model:
        fastest = min(
            metrics_by_model.items(),
            key=lambda x: x[1].get("latency_ms", float("inf")),
        )
        best_acc = max(
            metrics_by_model.items(),
            key=lambda x: x[1].get("accuracy", 0.0),
        )
        cheapest = min(
            metrics_by_model.items(),
            key=lambda x: x[1].get("cost_per_1m_tokens_usd", float("inf")),
        )
        most_efficient = max(
            metrics_by_model.items(),
            key=lambda x: x[1].get("efficiency_score", 0.0),
        )
        lines += [
            f"- **Lowest latency**: `{fastest[0]}` \u2014 "
            f"{fastest[1].get('latency_ms', 0):.1f} ms",
            f"- **Highest accuracy**: `{best_acc[0]}` \u2014 "
            f"{best_acc[1].get('accuracy', 0):.3f}",
            f"- **Lowest cost**: `{cheapest[0]}` \u2014 "
            f"${cheapest[1].get('cost_per_1m_tokens_usd', 0):.4f} / 1M tokens",
            f"- **Best efficiency** (accuracy \u00f7 cost): `{most_efficient[0]}`",
        ]

    lines += [
        "",
        "## Notes",
        "",
        "- Mock responses used when API keys or local services are unavailable.",
        "- Latencies are end-to-end wall-clock times including any network round-trip.",
        "- Energy estimates are approximate; actual consumption varies by hardware.",
        "- Benchmark results are written to `benchmark_results/` as "
        "HTML, CSV, JSON, and Markdown.",
        "",
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown summary \u2192 %s", output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Real-time dashboard (optional FastAPI endpoint)
# ---------------------------------------------------------------------------

def create_dashboard_app(
    results_json_path: Path | str = RESULTS_DIR / "results.json",
) -> Any:
    """Create a minimal FastAPI app with a live dashboard and JSON API endpoint.

    Requires:  pip install fastapi uvicorn
    Run with:  uvicorn benchmark_report:app --port 8001
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:
        raise ImportError(
            "fastapi is required for the dashboard. "
            "Install with: pip install fastapi uvicorn"
        ) from exc

    results_path = Path(results_json_path)
    app = FastAPI(title="OctoTetrahedral Benchmark Dashboard")

    _DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="10">
  <title>Benchmark Live Dashboard</title>
  <style>
    body{{font-family:monospace;background:#0d1117;color:#e6edf3;padding:24px}}
    h1{{color:#58a6ff}} a{{color:#58a6ff}} pre{{background:#161b22;
    border:1px solid #30363d;border-radius:6px;padding:16px;overflow:auto;
    max-height:70vh}}
  </style>
</head>
<body>
  <h1>\U0001f534 Live Benchmark Dashboard</h1>
  <p>Auto-refreshes every 10 s &nbsp;\u00b7&nbsp;
     <a href="/api/results">Raw JSON</a> &nbsp;\u00b7&nbsp;
     <a href="/report">HTML Report</a></p>
  <pre id="out">Loading\u2026</pre>
  <script>
    fetch("/api/results")
      .then(r => r.json())
      .then(d => document.getElementById("out").textContent =
            JSON.stringify(d, null, 2))
      .catch(() => document.getElementById("out").textContent =
              "No results yet \u2014 run benchmark_suite.py first.");
  </script>
</body>
</html>"""

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(content=_DASHBOARD_HTML)

    @app.get("/api/results", response_class=JSONResponse)
    async def api_results() -> JSONResponse:
        if results_path.exists():
            with results_path.open() as fh:
                data = json.load(fh)
            return JSONResponse(content=data)
        return JSONResponse(content={"status": "no results yet"}, status_code=404)

    return app
