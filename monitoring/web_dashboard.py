"""Web dashboard with FastAPI backend and Prometheus export.

Run directly::

    python -m monitoring.web_dashboard --port 8000

Then visit http://localhost:8000/ for the live dashboard, or
http://localhost:8000/metrics for Prometheus-format metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, PlainTextResponse, Response
    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False

from monitoring.config import MonitoringConfig
from monitoring.metrics_recorder import MetricsRecorder

# ---------------------------------------------------------------------------
# Inline HTML frontend (served at /)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Unified Cognitive Stack — Live Monitor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>
    :root { --bg:#0d1117; --fg:#c9d1d9; --accent:#58a6ff; --green:#3fb950;
            --yellow:#d29922; --red:#f85149; --card:#161b22; --border:#30363d; }
    body.light { --bg:#ffffff; --fg:#24292f; --card:#f6f8fa; --border:#d0d7de; }
    * { box-sizing:border-box; margin:0; padding:0; }
    body { background:var(--bg); color:var(--fg); font-family:monospace;
           font-size:14px; padding:16px; }
    h1 { color:var(--accent); margin-bottom:16px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
            gap:12px; margin-bottom:16px; }
    .card { background:var(--card); border:1px solid var(--border);
            border-radius:8px; padding:16px; }
    .card h2 { font-size:12px; text-transform:uppercase; color:var(--accent);
               margin-bottom:8px; }
    .big { font-size:2em; font-weight:bold; }
    .green { color:var(--green); } .yellow { color:var(--yellow); }
    .red { color:var(--red); }
    .chart-wrap { height:200px; position:relative; }
    table { width:100%; border-collapse:collapse; }
    td,th { padding:4px 8px; border-bottom:1px solid var(--border); text-align:left; }
    th { color:var(--accent); }
    #status { position:fixed; bottom:8px; right:12px; font-size:11px;
              color:var(--accent); }
    button { background:var(--card); border:1px solid var(--border);
             color:var(--fg); padding:4px 12px; border-radius:4px; cursor:pointer; }
  </style>
</head>
<body>
<h1>🧠 Unified Cognitive Stack — Live Monitor</h1>
<button onclick="document.body.classList.toggle('light')">🌓 Toggle Theme</button>

<div class="grid">
  <div class="card">
    <h2>Coherence</h2>
    <div id="coh-val" class="big green">—</div>
    <div style="margin-top:4px">avg: <span id="coh-avg">—</span></div>
  </div>
  <div class="card">
    <h2>Latency (ms)</h2>
    <div>p50: <span id="lat-p50" class="big">—</span></div>
    <div>p99: <span id="lat-p99">—</span>
         &nbsp;p99.9: <span id="lat-p999">—</span></div>
  </div>
  <div class="card">
    <h2>Throughput</h2>
    <div class="big"><span id="rps">—</span> <small>req/s</small></div>
    <div>total: <span id="total">—</span></div>
  </div>
  <div class="card">
    <h2>Limbs Active</h2>
    <div id="limb-bar" class="big" style="letter-spacing:2px">—</div>
  </div>
  <div class="card">
    <h2>Action Channel</h2>
    <div id="action-ch" class="big">—</div>
  </div>
  <div class="card">
    <h2>SLA Status</h2>
    <div id="sla" class="big green">—</div>
  </div>
</div>

<div class="grid">
  <div class="card" style="grid-column: span 2">
    <h2>Coherence over time</h2>
    <div class="chart-wrap"><canvas id="cohChart"></canvas></div>
  </div>
  <div class="card" style="grid-column: span 2">
    <h2>Latency over time (ms)</h2>
    <div class="chart-wrap"><canvas id="latChart"></canvas></div>
  </div>
</div>

<div class="card">
  <h2>Metrics Table</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th><th>SLA</th></tr></thead>
    <tbody id="metrics-table"></tbody>
  </table>
</div>

<div id="status">⏳ Connecting…</div>

<script>
const MAX_POINTS = 120;
const mkDataset = (label, color) => ({
  label, borderColor: color, backgroundColor: color + '33',
  data: [], tension: 0.3, pointRadius: 0, fill: false
});
const cohChart = new Chart(document.getElementById('cohChart'), {
  type: 'line',
  data: { labels: [], datasets: [mkDataset('Coherence', '#58a6ff')] },
  options: { animation: false, scales: { y: { min: 0, max: 1 } } }
});
const latChart = new Chart(document.getElementById('latChart'), {
  type: 'line',
  data: { labels: [], datasets: [
    mkDataset('p50', '#3fb950'), mkDataset('p99', '#d29922'),
    mkDataset('p99.9', '#f85149')
  ]},
  options: { animation: false }
});

function push(chart, index, values) {
  const ts = new Date().toLocaleTimeString();
  chart.data.labels.push(ts);
  values.forEach((v, i) => chart.data.datasets[i].data.push(v));
  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets.forEach(d => d.data.shift());
  }
  chart.update('none');
}

function colorClass(v, green, yellow) {
  return v >= green ? 'green' : v >= yellow ? 'yellow' : 'red';
}
function latClass(v) { return v < 20 ? 'green' : v < 50 ? 'yellow' : 'red'; }

function update(d) {
  const cur = d.current || {};
  const all = d.all || {};
  const coh = cur.coherence ?? 0;
  const latP50 = all.latency_p50 ?? cur.latency_ms ?? 0;
  const latP99 = all.latency_p99 ?? 0;
  const latP999 = all.latency_p999 ?? 0;

  const cohEl = document.getElementById('coh-val');
  cohEl.textContent = coh.toFixed(4);
  cohEl.className = 'big ' + colorClass(coh, 0.9, 0.8);
  document.getElementById('coh-avg').textContent = (all.coherence_mean ?? coh).toFixed(4);

  const latEl = document.getElementById('lat-p50');
  latEl.textContent = latP50.toFixed(1);
  latEl.className = 'big ' + latClass(latP50);
  document.getElementById('lat-p99').textContent = latP99.toFixed(1);
  document.getElementById('lat-p999').textContent = latP999.toFixed(1);

  document.getElementById('rps').textContent = (d.throughput_rps ?? 0).toFixed(1);
  document.getElementById('total').textContent = d.total_inferences ?? 0;

  const limbs = cur.limbs_active ?? 0;
  document.getElementById('limb-bar').textContent =
    '█'.repeat(limbs) + '░'.repeat(Math.max(0, 8 - limbs)) + ` ${limbs}/8`;
  document.getElementById('action-ch').textContent = cur.action_channel ?? '—';

  const slaEl = document.getElementById('sla');
  if (coh >= 0.9 && latP50 < 20) {
    slaEl.textContent = '● GREEN'; slaEl.className = 'big green';
  } else if (coh >= 0.8 && latP50 < 50) {
    slaEl.textContent = '● YELLOW'; slaEl.className = 'big yellow';
  } else {
    slaEl.textContent = '● RED'; slaEl.className = 'big red';
  }

  push(cohChart, cohChart.data.labels.length, [coh]);
  push(latChart, latChart.data.labels.length, [latP50, latP99, latP999]);

  const rows = [
    ['Coherence', coh.toFixed(4), coh >= 0.9 ? '✅' : coh >= 0.8 ? '⚠️' : '❌'],
    ['Latency p50 (ms)', latP50.toFixed(1), latP50 < 20 ? '✅' : latP50 < 50 ? '⚠️' : '❌'],
    ['Coupling', (cur.coupling_strength ?? 0).toFixed(4), '—'],
    ['Phase', (cur.phase ?? 0).toFixed(4), '—'],
    ['Bias', (cur.bias ?? 0).toFixed(4), '—'],
    ['Task Signal', cur.task_signal || '—', '—'],
  ];
  document.getElementById('metrics-table').innerHTML = rows.map(r =>
    `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('');
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/metrics`);
  ws.onopen = () => document.getElementById('status').textContent = '🟢 Connected';
  ws.onmessage = e => { try { update(JSON.parse(e.data)); } catch(_) {} };
  ws.onclose = () => {
    document.getElementById('status').textContent = '🔴 Reconnecting…';
    setTimeout(connect, 2000);
  };
}
connect();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Prometheus format helper
# ---------------------------------------------------------------------------

def build_prometheus_output(stats: dict[str, Any]) -> str:
    """Format stats as OpenMetrics 1.0 (Prometheus) text."""
    cur = stats.get("current", {})
    all_s = stats.get("all", {})
    lines = [
        "# HELP unified_coherence Current coherence value",
        "# TYPE unified_coherence gauge",
        f'unified_coherence{{quantile="current"}} {cur.get("coherence", 0):.6f}',
        f'unified_coherence{{quantile="mean"}} {all_s.get("coherence_mean", 0):.6f}',
        "",
        "# HELP unified_latency_ms Inference latency in milliseconds",
        "# TYPE unified_latency_ms summary",
        f'unified_latency_ms{{quantile="p50"}} {all_s.get("latency_p50", 0):.4f}',
        f'unified_latency_ms{{quantile="p99"}} {all_s.get("latency_p99", 0):.4f}',
        f'unified_latency_ms{{quantile="p99.9"}} {all_s.get("latency_p999", 0):.4f}',
        "",
        "# HELP unified_limbs_active Number of active cognitive limbs",
        "# TYPE unified_limbs_active gauge",
        f'unified_limbs_active {cur.get("limbs_active", 0)}',
        "",
        "# HELP unified_inference_count Total number of inferences recorded",
        "# TYPE unified_inference_count counter",
        f'unified_inference_count {stats.get("total_inferences", 0)}',
        "",
        "# HELP unified_throughput_rps Inference throughput in requests per second",
        "# TYPE unified_throughput_rps gauge",
        f'unified_throughput_rps {stats.get("throughput_rps", 0):.4f}',
        "# EOF",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(
    recorder: MetricsRecorder | None = None,
    config: MonitoringConfig | None = None,
) -> "FastAPI":
    """Create and return the FastAPI application."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError("fastapi is required. Install it with: pip install fastapi")

    _config = config or MonitoringConfig()
    _recorder = recorder or MetricsRecorder(config=_config)

    app = FastAPI(title="Unified Cognitive Stack Monitor")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store references on the app for test access
    app.state.recorder = _recorder
    app.state.config = _config

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(_DASHBOARD_HTML)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "timestamp": time.time()}

    @app.get("/api/metrics/current")
    async def metrics_current() -> dict[str, Any]:
        return _recorder.get_rolling_stats()

    @app.get("/api/metrics/history")
    async def metrics_history(minutes: int = 5) -> dict[str, Any]:
        cutoff = time.time() - minutes * 60
        inferences = [
            r for r in _recorder.get_all_inferences()
            if r["timestamp"] >= cutoff
        ]
        return {"minutes": minutes, "count": len(inferences), "data": inferences}

    @app.get("/api/metrics/export")
    async def metrics_export(format: str = "json") -> Response:
        stats = _recorder.get_rolling_stats()
        if format == "prometheus":
            text = build_prometheus_output(stats)
            return PlainTextResponse(text, media_type="text/plain; version=0.0.4")
        return Response(
            content=json.dumps(stats), media_type="application/json"
        )

    @app.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        stats = _recorder.get_rolling_stats()
        return PlainTextResponse(
            build_prometheus_output(stats),
            media_type="text/plain; version=0.0.4",
        )

    @app.websocket("/ws/metrics")
    async def ws_metrics(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                stats = _recorder.get_rolling_stats()
                await websocket.send_json(stats)
                await asyncio.sleep(_config.web_update_frequency_sec)
        except WebSocketDisconnect:
            pass

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Web dashboard for UnifiedForwardModel metrics"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--history-minutes", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install uvicorn")
        return
    args = _parse_args(argv)
    config = MonitoringConfig(
        web_port=args.port,
        web_history_minutes=args.history_minutes,
    )
    app = create_app(config=config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
