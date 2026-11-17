"""Local web UI — drag a CSV in, get a ranked parameter sweep back."""

from __future__ import annotations

import base64
import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import data, licensing, search
from .strategies import FREE_STRATEGY, STRATEGIES
from .version import __version__

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SweepLab</title>
<style>
:root{--bg:#0b0f18;--panel:#121a29;--line:#22304a;--fg:#e8eef8;--muted:#8b9ab1;--accent:#38bdf8}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:44px 24px}
h1{font-size:26px;margin:0 0 4px} .sub{color:var(--muted);font-size:13px}
.drop{border:1.5px dashed var(--line);border-radius:14px;padding:34px;text-align:center;margin-top:22px;background:var(--panel)}
.drop.on{border-color:var(--accent)}
input[type=file]{display:none} label{color:var(--accent);cursor:pointer;font-weight:600}
select,button{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:9px;padding:9px 12px;font-size:13px}
button{background:var(--accent);color:#05121c;font-weight:650;border:none;cursor:pointer;padding:10px 20px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:18px}
pre{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;overflow:auto;font-size:12px;margin-top:18px;white-space:pre-wrap}
.err{color:#f87171} .tier{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.lock{border:1px dashed var(--line);border-radius:12px;padding:16px;margin-top:16px;font-size:13px}
.lock a{color:var(--accent)}
</style></head><body><div class="wrap">
<h1>SweepLab</h1><div class="sub">Find the parameter set that survives out of sample.</div>
<div class="tier">tier: __TIER__ · v__VERSION__</div>
<div class="drop" id="drop"><input type="file" id="file" accept=".csv,.tsv,.txt,.json">
<div>Drop a CSV/TSV with OHLCV columns, or <label for="file">browse</label></div>
<div class="sub" id="fname">no file selected</div></div>
<div class="row"><select id="strategy">__STRATEGIES__</select>
<select id="mode"><option value="grid">grid</option><option value="tpe">tpe (Pro)</option><option value="random">random (Pro)</option></select>
<button id="go">Run sweep</button></div>
__LOCK__
<pre id="out">results appear here</pre>
</div><script>
const drop=document.getElementById('drop'),file=document.getElementById('file'),out=document.getElementById('out');
let payload=null;
function read(f){const r=new FileReader();r.onload=()=>{payload=r.result.split(',')[1]||r.result;
document.getElementById('fname').textContent=f.name;};r.readAsDataURL(f);}
file.onchange=e=>{if(e.target.files[0])read(e.target.files[0])};
['dragover','dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();
drop.classList.toggle('on',ev==='dragover');if(ev==='drop'&&e.dataTransfer.files[0])read(e.dataTransfer.files[0]);}));
document.getElementById('go').onclick=async()=>{if(!payload){out.textContent='pick a file first';return;}
out.textContent='running…';
const res=await fetch('/api/sweep',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({data:payload,strategy:document.getElementById('strategy').value,mode:document.getElementById('mode').value})});
const j=await res.json();out.textContent=j.ok?j.report:'error: '+j.error;};
</script></body></html>"""


def _page() -> str:
    pro = licensing.is_pro()
    options = "".join(
        f'<option value="{name}">{spec["label"]}{"" if not spec["pro_only"] or pro else " (Pro)"}</option>'
        for name, spec in STRATEGIES.items()
    )
    lock = "" if pro else ('<div class="lock">Free tier: grid search on SMA cross. '
                           'Pro unlocks TPE search, all four strategies, walk-forward and the stability map. '
                           '<a href="https://sweeplab.gumroad.com">Get Pro</a></div>')
    return (PAGE.replace("__TIER__", "pro" if pro else "free")
            .replace("__VERSION__", __version__)
            .replace("__STRATEGIES__", options)
            .replace("__LOCK__", lock))


def _run(body: dict) -> dict:
    pro = licensing.is_pro()
    strategy = body.get("strategy", FREE_STRATEGY)
    mode = body.get("mode", "grid")
    if strategy not in STRATEGIES:
        return {"ok": False, "error": f"unknown strategy {strategy}"}
    if not pro and (strategy != FREE_STRATEGY or mode != "grid"):
        return {"ok": False, "error": "this option requires SweepLab Pro"}

    raw = body.get("data", "")
    blob = base64.b64decode(raw.split(",")[-1])
    tmp = Path(tempfile.mkdtemp()) / "upload.csv"
    tmp.write_bytes(blob)
    bars = data.load_bars(tmp)

    result = search.run_search(strategy, bars, mode=mode, n_trials=40 if mode != "grid" else 60)
    board = search.leaderboard(result, top=10).to_string(index=False)
    meta = data.describe(bars)
    return {"ok": True, "report": (
        f"{strategy} · {mode} · {result['n_trials']} trials · {meta['rows']} bars "
        f"({meta['start'][:10]} → {meta['end'][:10]})\n\n{board}\n\nbest: "
        + ", ".join(f"{k}={v:g}" for k, v in result["best_params"].items()))}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code: int, body: str, ctype: str = "text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body.encode()))
                         if isinstance(body, str) else str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode() if isinstance(body, str) else body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, _page())
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/sweep":
            self._send(404, "not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            payload = _run(body)
            status = 200 if payload.get("ok") else 400
        except Exception as exc:  # keep the UI alive on bad input
            payload = {"ok": False, "error": str(exc)}
            status = 400
        self._send(status, json.dumps(payload), "application/json")


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"SweepLab UI on http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
