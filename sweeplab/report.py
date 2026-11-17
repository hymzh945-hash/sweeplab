"""Single-file HTML report: best parameters, walk-forward decay, stability map."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from . import charts

CSS = """
:root{--bg:#0b0f18;--panel:#121a29;--panel2:#0f1624;--line:#22304a;--fg:#e8eef8;
--muted:#8b9ab1;--accent:#38bdf8;--good:#34d399;--warn:#fbbf24;--bad:#f87171}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:34px 26px 60px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
margin:34px 0 12px;font-weight:600}
.sub{color:var(--muted);font-size:13px}
.meta{display:flex;flex-wrap:wrap;gap:10px 22px;margin-top:14px;font-size:12px;color:var(--muted)}
.meta b{color:var(--fg);font-weight:600}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.card .l{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.card .v{font-size:23px;font-weight:650;margin-top:6px;letter-spacing:-.02em}
.card .f{font-size:11px;color:var(--muted);margin-top:2px}
.pos{color:var(--good)} .neg{color:var(--bad)} .neu{color:var(--accent)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.chip{background:var(--panel2);border:1px solid var(--line);border-radius:999px;
padding:6px 13px;font-size:12px;font-variant-numeric:tabular-nums}
.chip b{color:var(--accent);font-weight:600}
img{width:100%;border-radius:12px;border:1px solid var(--line);display:block;margin-top:8px}
table{width:100%;border-collapse:collapse;margin-top:8px;font-size:12.5px;
font-variant-numeric:tabular-nums}
th{text-align:left;color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
letter-spacing:.06em;border-bottom:1px solid var(--line);padding:8px 10px}
td{padding:8px 10px;border-bottom:1px solid #1a2436}
tr.best td{background:rgba(56,189,248,.07)}
.verdict{border-radius:12px;padding:16px 18px;margin-top:14px;border:1px solid var(--line);
background:var(--panel);display:flex;gap:14px;align-items:flex-start}
.verdict .bar{width:4px;border-radius:2px;flex:none;height:44px}
.verdict .t{font-size:16px;font-weight:650}
.verdict .d{color:var(--muted);font-size:12.5px;margin-top:4px}
.lock{background:linear-gradient(135deg,#131c2c,#0d1420);border:1px dashed var(--line);
border-radius:14px;padding:22px;text-align:center;margin-top:12px}
.lock .t{font-weight:650;font-size:15px}
.lock .d{color:var(--muted);font-size:12.5px;margin:6px auto 12px;max-width:520px}
.lock a{display:inline-block;background:var(--accent);color:#05121c;text-decoration:none;
font-weight:650;padding:9px 18px;border-radius:9px;font-size:13px}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);
font-size:11.5px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
"""


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _num(value: float, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _img(b64: str, alt: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">'


def _table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        cls = ' class="best"' if i == 0 else ""
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        rows.append(f"<tr{cls}>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build(
    result: Dict,
    bars_meta: Dict,
    robust_result: Optional[Dict] = None,
    stability: Optional[Dict] = None,
    version: str = "0.1.0",
    pro: bool = True,
    symbol: str = "",
) -> str:
    best = result["best_metrics"]
    params = result["best_params"]
    trial_df = result["trials"]

    kpis = [
        ("Best Sharpe", _num(best["sharpe"]), "in-sample", "neu"),
        ("Total return", _pct(best["total_return"]), "net of costs", "pos" if best["total_return"] > 0 else "neg"),
        ("Max drawdown", _pct(best["max_drawdown"]), "peak to trough", "neg"),
        ("Trades", str(best["trades"]), "round trips", ""),
        ("Exposure", _pct(best["exposure"]), "time in market", ""),
        ("Trials", str(result["n_trials"]), f"mode {result['mode']}", ""),
    ]
    cards = "".join(
        f'<div class="card"><div class="l">{l}</div><div class="v {cls}">{v}</div>'
        f'<div class="f">{f}</div></div>' for l, v, f, cls in kpis
    )
    chips = "".join(f'<span class="chip">{html.escape(k)} <b>{v:g}</b></span>'
                    if isinstance(v, (int, float)) else
                    f'<span class="chip">{html.escape(k)} <b>{html.escape(str(v))}</b></span>'
                    for k, v in params.items())

    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SweepLab report — {html.escape(result['strategy'])}</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>SweepLab — parameter sweep report</h1>
<div class="sub">{html.escape(result['strategy'])} · optimising <b>{html.escape(result['metric'])}</b>
over {result['n_trials']} trials</div>
<div class="meta">
<span>bars <b>{bars_meta['rows']}</b></span>
<span>window <b>{html.escape(bars_meta['start'][:10])} → {html.escape(bars_meta['end'][:10])}</b></span>
{f'<span>symbol <b>{html.escape(symbol)}</b></span>' if symbol else ''}
<span>generated <b>{datetime.now():%Y-%m-%d %H:%M}</b></span>
</div>
<div class="cards">{cards}</div>
<h2>Winning parameters</h2>
<div class="chips">{chips}</div>
""",
    _img(charts.equity_curve(best["_equity"]), "equity curve"),
    "</div>"]

    if robust_result:
        s = robust_result["summary"]
        rows = robust_result["table"].copy()
        rows["is_return"] = (rows["is_return"] * 100).round(1).astype(str) + "%"
        rows["oos_return"] = (rows["oos_return"] * 100).round(1).astype(str) + "%"
        rows["oos_max_dd"] = (rows["oos_max_dd"] * 100).round(1).astype(str) + "%"
        rows = rows.rename(columns={"fold": "#", "is_sharpe": "IS Sharpe", "oos_sharpe": "OOS Sharpe",
                                    "is_return": "IS ret", "oos_return": "OOS ret",
                                    "oos_max_dd": "OOS DD", "decay": "decay"})
        parts.append(f"""<h2>Out-of-sample survival</h2>
<div class="verdict"><div class="bar" style="background:{s['verdict_color']}"></div>
<div><div class="t" style="color:{s['verdict_color']}">{html.escape(s['verdict'])}</div>
<div class="d">Mean in-sample Sharpe {_num(s['is_sharpe_mean'])} → out-of-sample {_num(s['oos_sharpe_mean'])}.
Positive OOS in {s['oos_positive_folds']}/{s['folds']} folds · worst OOS drawdown {_pct(s['worst_oos_dd'])}.</div></div></div>
{_table(rows[['#', 'train_end', 'test_start', 'test_end', 'IS Sharpe', 'OOS Sharpe', 'IS ret', 'OOS ret', 'OOS DD', 'params']])}
""")
        parts.append(_img(charts.walkforward_bars(robust_result["table"]), "walk-forward bars"))
    elif not pro:
        parts.append("""<h2>Out-of-sample survival</h2>
<div class="lock"><div class="t">Walk-forward audit is a Pro feature</div>
<div class="d">Pro re-optimises on each train slice and trades the winner on the next slice,
so you can see whether the parameters survive — or were just fitted to the past.</div>
<a href="https://sweeplab.gumroad.com">Get SweepLab Pro</a></div>""")

    if stability:
        st = stability
        parts.append(f"""<h2>Stability map — {html.escape(st['axes'][0])} × {html.escape(st['axes'][1])}</h2>
<div class="meta"><span>best point <b>{st['best']['x']:g} / {st['best']['y']:g} → {st['best']['value']:.2f}</b></span>
<span>neighbour mean <b>{st['neighbour_mean']:.2f}</b></span>
<span>plateau ratio <b>{st['plateau_ratio']:.2f}</b></span>
<span><b>{html.escape(st['verdict'])}</b></span></div>
""")
        parts.append(_img(charts.heatmap(st["matrix"], st["x_values"], st["y_values"],
                                        st["axes"][0], st["axes"][1]), "stability map"))
    elif not pro:
        parts.append("""<h2>Stability map</h2>
<div class="lock"><div class="t">Neighbourhood check is a Pro feature</div>
<div class="d">The best point in a sweep is usually a spike. Pro scores the cells around it:
a plateau means the edge is real, a spike means you fitted noise.</div>
<a href="https://sweeplab.gumroad.com">Get SweepLab Pro</a></div>""")

    lb = trial_df.head(10).copy()
    keep = list(result["space"].keys()) + ["score", "total_return", "max_drawdown", "trades"]
    lb = lb[[c for c in keep if c in lb.columns]].round(4)
    parts.append(f"<h2>Leaderboard (top 10 of {result['n_trials']})</h2>{_table(lb)}")
    parts.append(f"""<footer><span>SweepLab {version} · built on Optuna (MIT)</span>
<span>Past performance is not indicative of future results.</span></footer></div></body></html>""")
    return "".join(parts)
