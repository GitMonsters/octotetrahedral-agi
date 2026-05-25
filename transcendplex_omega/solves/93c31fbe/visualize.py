import json, sys
sys.path.insert(0, ".")
from solver import solve

task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/93c31fbe.json"))
colors = ['#000','#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25']

def grid_html(g, label):
    h, w = len(g), len(g[0])
    s = f'<div style="display:inline-block;margin:8px;vertical-align:top"><div style="font-weight:bold;margin-bottom:4px">{label}</div><table style="border-collapse:collapse">'
    for row in g:
        s += '<tr>'
        for v in row:
            s += f'<td style="width:16px;height:16px;background:{colors[v]};border:1px solid #333"></td>'
        s += '</tr>'
    s += '</table></div>'
    return s

html = '<html><body style="background:#222;color:#fff;font-family:monospace">'
html += '<h1>ARC 93c31fbe — Frame Reflection</h1>'
html += '<p>Frames (L-shaped corners) contain blue patterns. Wider frames → left-right mirror. Taller frames → top-bottom mirror. Loose blue pixels removed.</p>'

for i, p in enumerate(task['train']):
    result = solve(p['input'])
    ok = result == p['output']
    html += f'<h2>Train {i} {"✅" if ok else "❌"}</h2><div>'
    html += grid_html(p['input'], 'Input')
    html += grid_html(p['output'], 'Expected')
    html += grid_html(result, 'Solver')
    html += '</div>'

for i, p in enumerate(task['test']):
    result = solve(p['input'])
    ok = result == p.get('output', result)
    html += f'<h2>Test {i} {"✅" if ok else "❌"}</h2><div>'
    html += grid_html(p['input'], 'Input')
    if 'output' in p:
        html += grid_html(p['output'], 'Expected')
    html += grid_html(result, 'Solver Output')
    html += '</div>'

html += '</body></html>'
with open('visualization.html', 'w') as f:
    f.write(html)
print("Wrote visualization.html")
