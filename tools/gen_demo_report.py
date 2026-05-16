"""
Slick simulation demo report generator.
Runs N forward passes, collects per-pass metrics, and writes a rich HTML dashboard.
"""
import argparse, subprocess, sys, os, json
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import OctoTetrahedralModel


def run_demo(n_passes: int = 60, out: str = "logs/cohesion/demo.html", open_browser: bool = False) -> None:
    print(f"Loading OctoTetrahedralModel…")
    m = OctoTetrahedralModel()
    total_params = sum(p.numel() for p in m.parameters()) / 1e6

    timeline: list[dict] = []
    print(f"Running {n_passes}-pass simulation…")
    for i in range(n_passes):
        x = torch.randint(0, 1000, (1, 16))
        out_dict = m(x)
        ci = out_dict["cohesion_info"]
        bs = ci["braid_stats"]
        path = out_dict.get("two_speed_info", {}).get("path", "slow")
        timeline.append({
            "pass": i + 1,
            "cohesion": round(ci["cohesion_score"], 4),
            "ewma": round(ci["ewma_score"], 4),
            "limb_balance": round(ci["limb_balance"], 4),
            "skill_coverage": round(ci["skill_coverage"], 4),
            "latency": round(ci["latency_score"], 4),
            "limbs_active": ci["limbs_active"],
            "events": bs["total_events"],
            "path": path,
        })

    final = m.cohesion_score()
    bs_f = final["braid_stats"]

    # Limb activation counts
    from core.cognitive_cohesion_braid import ALL_LIMBS
    limb_counts = {l: m.cohesion_braid.scorer.limb_counts.get(l, 0) for l in ALL_LIMBS}

    # Skill firing counts
    from core.cognitive_cohesion_braid import SKILL_LIMB_MAP, SKILL_SOURCE
    skill_counts = {s: m.cohesion_braid.scorer.skill_counts.get(s, 0) for s in SKILL_LIMB_MAP}

    # ── HTML ────────────────────────────────────────────────────────────────────
    tl_json   = json.dumps(timeline)
    lc_labels = json.dumps(list(limb_counts.keys()))
    lc_vals   = json.dumps(list(limb_counts.values()))
    sk_labels = json.dumps(list(skill_counts.keys()))
    sk_vals   = json.dumps(list(skill_counts.values()))

    cohesion_pct = final["cohesion_score"] * 100
    ewma_pct     = final["ewma_score"] * 100
    lb_pct       = final["limb_balance"] * 100
    sc_pct       = final["skill_coverage"] * 100

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OctoTetrahedral AGI — Cognitive Cohesion Demo</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg:#080e1e; --card:#0d1730; --border:#1c2e52;
  --accent:#5b9cf6; --accent2:#7ee8a2; --accent3:#f6a05b;
  --text:#dce9ff; --muted:#6b88b5;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,Segoe UI,Roboto,sans-serif;padding:2rem}}
h1{{font-size:1.6rem;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.2rem}}
.sub{{color:var(--muted);font-size:.85rem;margin-bottom:1.8rem}}
.grid{{display:grid;gap:1rem}}
.g4{{grid-template-columns:repeat(4,1fr)}}
.g2{{grid-template-columns:repeat(2,1fr)}}
.g3{{grid-template-columns:repeat(3,1fr)}}
@media(max-width:900px){{.g4,.g3{{grid-template-columns:repeat(2,1fr)}}.g2{{grid-template-columns:1fr}}}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.2rem}}
.kpi-val{{font-size:2rem;font-weight:700;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.kpi-lbl{{font-size:.78rem;color:var(--muted);margin-top:.15rem;text-transform:uppercase;letter-spacing:.05em}}
.bar-wrap{{height:8px;background:#12203a;border-radius:4px;overflow:hidden;margin-top:.6rem}}
.bar-fill{{height:100%;border-radius:4px;transition:width .8s ease}}
.b1{{background:linear-gradient(90deg,#5b9cf6,#7ee8a2)}}
.b2{{background:linear-gradient(90deg,#f6a05b,#f6d35b)}}
.b3{{background:linear-gradient(90deg,#c05bf6,#5bf6e8)}}
.b4{{background:linear-gradient(90deg,#f65b5b,#f6a05b)}}
h2{{font-size:1rem;color:var(--accent);margin-bottom:.9rem;text-transform:uppercase;letter-spacing:.06em}}
.badge{{display:inline-block;background:#162041;border:1px solid var(--border);border-radius:6px;padding:.2rem .55rem;font-size:.78rem;margin:.2rem}}
.badge span{{color:var(--accent2);font-weight:700}}
canvas{{max-height:260px}}
.flow{{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin-top:.5rem}}
.fnode{{background:#142042;border:1px solid var(--border);border-radius:8px;padding:.4rem .8rem;font-size:.82rem}}
.farrow{{color:var(--muted)}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th,td{{padding:.35rem .6rem;border-bottom:1px solid var(--border);text-align:left}}
th{{color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}}
.hi{{color:var(--accent2)}} .lo{{color:var(--muted)}}
</style>
</head>
<body>

<h1>🧬 OctoTetrahedral AGI — Cognitive Cohesion Braid</h1>
<p class="sub">Live simulation demo · {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · {total_params:.1f}M parameters · {n_passes} passes</p>

<!-- KPI row -->
<div class="grid g4" style="margin-bottom:1rem">
  <div class="card">
    <div class="kpi-val">{final['cohesion_score']:.3f}</div>
    <div class="kpi-lbl">Cohesion Score</div>
    <div class="bar-wrap"><div class="bar-fill b1" style="width:{cohesion_pct:.1f}%"></div></div>
  </div>
  <div class="card">
    <div class="kpi-val">{final['ewma_score']:.3f}</div>
    <div class="kpi-lbl">EWMA Score</div>
    <div class="bar-wrap"><div class="bar-fill b2" style="width:{ewma_pct:.1f}%"></div></div>
  </div>
  <div class="card">
    <div class="kpi-val">{final['limb_balance']:.3f}</div>
    <div class="kpi-lbl">Limb Balance (entropy)</div>
    <div class="bar-wrap"><div class="bar-fill b3" style="width:{lb_pct:.1f}%"></div></div>
  </div>
  <div class="card">
    <div class="kpi-val">{final['skill_coverage']:.3f}</div>
    <div class="kpi-lbl">Skill Coverage</div>
    <div class="bar-wrap"><div class="bar-fill b4" style="width:{sc_pct:.1f}%"></div></div>
  </div>
</div>

<!-- Badges row -->
<div class="card" style="margin-bottom:1rem">
  <h2>System Overview</h2>
  <div style="display:flex;flex-wrap:wrap;gap:.4rem">
    <div class="badge">Params <span>{total_params:.1f}M</span></div>
    <div class="badge">Passes <span>{n_passes}</span></div>
    <div class="badge">Limbs Active <span>{final['limbs_active']}/{len(ALL_LIMBS)}</span></div>
    <div class="badge">Total Events <span>{bs_f['total_events']}</span></div>
    <div class="badge">Latency Score <span>{final['latency_score']:.3f}</span></div>
    <div class="badge">SIMULA→EUPHAN <span>{bs_f['simula_to_euphan']}</span></div>
    <div class="badge">EUPHAN→HERMES <span>{bs_f['euphan_to_hermes']}</span></div>
    <div class="badge">HERMES→SIMULA <span>{bs_f['hermes_to_simula']}</span></div>
  </div>
  <!-- Braid flow -->
  <div class="flow" style="margin-top:1rem">
    <div class="fnode">SIMULA<br><small style="color:var(--muted)">Descriptive</small></div>
    <div class="farrow">→ {bs_f['simula_to_euphan']} →</div>
    <div class="fnode">EUPHAN<br><small style="color:var(--muted)">Predictive</small></div>
    <div class="farrow">→ {bs_f['euphan_to_hermes']} →</div>
    <div class="fnode">HERMES<br><small style="color:var(--muted)">Prescriptive</small></div>
    <div class="farrow">→ {bs_f['hermes_to_simula']} →</div>
    <div class="fnode" style="border-color:var(--accent)">SIMULA<br><small style="color:var(--accent)">↺ closed loop</small></div>
  </div>
</div>

<!-- Charts row 1 -->
<div class="grid g2" style="margin-bottom:1rem">
  <div class="card">
    <h2>Cohesion Score Timeline</h2>
    <canvas id="timelineChart"></canvas>
  </div>
  <div class="card">
    <h2>Limb Activation Counts</h2>
    <canvas id="limbChart"></canvas>
  </div>
</div>

<!-- Charts row 2 -->
<div class="grid g2" style="margin-bottom:1rem">
  <div class="card">
    <h2>EWMA vs Instantaneous Cohesion</h2>
    <canvas id="ewmaChart"></canvas>
  </div>
  <div class="card">
    <h2>Skill Firing Frequency</h2>
    <canvas id="skillChart"></canvas>
  </div>
</div>

<!-- Limb table -->
<div class="card" style="margin-bottom:1rem">
  <h2>Limb Activation Detail</h2>
  <table>
    <tr><th>Limb</th><th>Activations</th><th>Share</th></tr>
    {"".join(
        f"<tr><td>{l}</td>"
        f"<td class=\"{'hi' if limb_counts[l]>0 else 'lo'}\">{limb_counts[l]}</td>"
        f"<td><div class='bar-wrap' style='width:100px;display:inline-block'>"
        f"<div class='bar-fill b1' style='width:{limb_counts[l]/max(max(limb_counts.values()),1)*100:.0f}%'></div></div></td></tr>"
        for l in ALL_LIMBS
    )}
  </table>
</div>

<script>
const tl = {tl_json};
const passes = tl.map(d=>d.pass);
const cohesionVals = tl.map(d=>d.cohesion);
const ewmaVals = tl.map(d=>d.ewma);

const GRID = {{ color:'rgba(28,46,82,.6)' }};
const TICK = {{ color:'#6b88b5', font:{{size:11}} }};
const baseOpts = {{
  responsive:true, animation:{{duration:600}},
  plugins:{{ legend:{{ labels:{{ color:'#dce9ff', font:{{size:11}} }} }} }},
  scales:{{
    x:{{ grid:GRID, ticks:TICK }},
    y:{{ grid:GRID, ticks:TICK }}
  }}
}};

// Timeline
new Chart(document.getElementById('timelineChart'), {{
  type:'line',
  data:{{ labels:passes,
    datasets:[{{ label:'Cohesion Score', data:cohesionVals, borderColor:'#5b9cf6', backgroundColor:'rgba(91,156,246,.12)', fill:true, tension:.35, pointRadius:2 }}]
  }},
  options:{{ ...baseOpts, scales:{{ x:{{ ...baseOpts.scales.x, title:{{ display:true,text:'Pass',color:'#6b88b5' }} }}, y:{{ ...baseOpts.scales.y, min:0,max:1 }} }} }}
}});

// EWMA
new Chart(document.getElementById('ewmaChart'), {{
  type:'line',
  data:{{ labels:passes,
    datasets:[
      {{ label:'Instantaneous', data:cohesionVals, borderColor:'#5b9cf6', backgroundColor:'rgba(91,156,246,.08)', fill:false, tension:.3, pointRadius:1 }},
      {{ label:'EWMA', data:ewmaVals, borderColor:'#7ee8a2', backgroundColor:'rgba(126,232,162,.12)', fill:true, tension:.5, pointRadius:0, borderWidth:2 }}
    ]
  }},
  options:{{ ...baseOpts, scales:{{ x:{{ ...baseOpts.scales.x }}, y:{{ ...baseOpts.scales.y, min:0,max:1 }} }} }}
}});

// Limbs
const limbLabels = {lc_labels};
const limbVals   = {lc_vals};
new Chart(document.getElementById('limbChart'), {{
  type:'bar',
  data:{{ labels:limbLabels,
    datasets:[{{ label:'Activations', data:limbVals,
      backgroundColor:limbLabels.map((_,i)=>`hsl(${{200+i*14}},70%,55%)`),
      borderRadius:5 }}]
  }},
  options:{{ ...baseOpts, plugins:{{ legend:{{ display:false }} }},
    scales:{{ x:{{ ...baseOpts.scales.x, ticks:{{ ...TICK, maxRotation:45 }} }}, y:{{ ...baseOpts.scales.y }} }} }}
}});

// Skills
const skLabels = {sk_labels};
const skVals   = {sk_vals};
new Chart(document.getElementById('skillChart'), {{
  type:'bar',
  data:{{ labels:skLabels,
    datasets:[{{ label:'Fires', data:skVals,
      backgroundColor:skLabels.map((_,i)=>`hsl(${{130+i*18}},65%,55%)`),
      borderRadius:5 }}]
  }},
  options:{{ ...baseOpts, plugins:{{ legend:{{ display:false }} }},
    scales:{{ x:{{ ...baseOpts.scales.x, ticks:{{ ...TICK, maxRotation:55 }} }}, y:{{ ...baseOpts.scales.y }} }} }}
}});
</script>

</body></html>"""

    Path(os.path.dirname(out) or ".").mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html)
    print(f"\n✅ Report written → {out}")
    print(f"   cohesion_score : {final['cohesion_score']:.4f}")
    print(f"   EWMA           : {final['ewma_score']:.4f}")
    print(f"   limbs_active   : {final['limbs_active']}/{len(ALL_LIMBS)}")
    print(f"   total_events   : {bs_f['total_events']}")

    if open_browser:
        subprocess.run(["open", out])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OctoTetrahedral AGI cohesion sim demo")
    p.add_argument("--passes", type=int, default=60)
    p.add_argument("--out",    default="logs/cohesion/demo.html")
    p.add_argument("--open",   action="store_true")
    args = p.parse_args()
    run_demo(args.passes, args.out, args.open)
