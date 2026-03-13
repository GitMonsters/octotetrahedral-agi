import json

COLORS = ['#000','#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25']

with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/b0f4d537.json") as f:
    task = json.load(f)

from solver import solve

def grid_html(g, label=""):
    h = f'<div style="display:inline-block;margin:8px;vertical-align:top"><div style="font-weight:bold;margin-bottom:4px">{label}</div><table style="border-collapse:collapse">'
    for row in g:
        h += '<tr>'
        for v in row:
            h += f'<td style="width:28px;height:28px;background:{COLORS[v]};border:1px solid #444;text-align:center;color:#fff;font-size:11px">{v}</td>'
        h += '</tr>'
    h += '</table></div>'
    return h

html = '<html><body style="background:#1a1a2e;color:#fff;font-family:monospace;padding:20px">'
html += '<h1>ARC Task b0f4d537</h1>'

for i, p in enumerate(task["train"]):
    res = solve(p["input"])
    ok = res == p["output"]
    html += f'<h2>Train {i} {"✅" if ok else "❌"}</h2><div>'
    html += grid_html(p["input"], "Input")
    html += grid_html(p["output"], "Expected")
    html += grid_html(res, "Predicted")
    html += '</div>'

for i, p in enumerate(task["test"]):
    res = solve(p["input"])
    ok = "output" in p and res == p["output"]
    html += f'<h2>Test {i} {"✅" if ok else "🔮"}</h2><div>'
    html += grid_html(p["input"], "Input")
    if "output" in p:
        html += grid_html(p["output"], "Expected")
    html += grid_html(res, "Predicted")
    html += '</div>'

html += '</body></html>'

with open("/Users/evanpieser/arc-puzzle-catalog/solves/b0f4d537/b0f4d537.html", "w") as f:
    f.write(html)
print("Wrote b0f4d537.html")
