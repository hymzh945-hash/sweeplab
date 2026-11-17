"""Command line interface — `sweeplab <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import data, licensing, report, robust, search
from .strategies import STRATEGIES, list_strategies
from .version import __version__

FREE_STRATEGY = "sma_cross"
UPLOAD_URL = "https://sweeplab.gumroad.com"


def _pro() -> bool:
    return licensing.is_pro()


def _require_pro(feature: str) -> None:
    print(f"{feature} is part of SweepLab Pro — {UPLOAD_URL}")
    print("Already bought? drop license.json in ~/sweeplab-license.json or set SWEEPLAB_LICENSE.")
    raise SystemExit(1)


def _parse_params(text: str | None) -> dict:
    if not text:
        return {}
    out = {}
    for chunk in text.split(","):
        if not chunk.strip():
            continue
        key, _, value = chunk.partition("=")
        out[key.strip()] = float(value) if "." in value else int(value)
    return out


def cmd_strategies(args: argparse.Namespace) -> int:
    print(f"SweepLab {__version__} · {len(STRATEGIES)} strategy templates\n")
    for item in list_strategies():
        mark = "free" if not item["pro_only"] else "Pro"
        space = ", ".join(f"{k}{v}" for k, v in item["space"].items())
        print(f"  {item['name']:<16} [{mark:<4}] {item['label']}")
        print(f"  {'':<16}  space: {space}")
    return 0


def _load(args) -> pd.DataFrame:
    if args.input:
        bars = data.load_bars(args.input)
    else:
        bars = data.synthetic_bars(rows=getattr(args, "rows", 1200))
        print("no --input given, using synthetic bars")
    if len(bars) < 120:
        print("warning: fewer than 120 bars, results will be noisy")
    return bars


def cmd_sweep(args: argparse.Namespace) -> int:
    pro = _pro()
    if not pro:
        if args.strategy != FREE_STRATEGY:
            _require_pro(f"Strategy '{args.strategy}'")
        if args.mode != "grid":
            _require_pro(f"Search mode '{args.mode}'")
    if args.strategy not in STRATEGIES:
        print(f"unknown strategy {args.strategy!r}; see `sweeplab strategies`")
        return 2

    bars = _load(args)
    space = None
    if args.grid:
        space = {}
        for chunk in args.grid.split(";"):
            if not chunk.strip():
                continue
            key, _, values = chunk.partition("=")
            space[key.strip()] = [float(v) if "." in v else int(v) for v in values.split(",")]

    result = search.run_search(
        args.strategy, bars, space=space, mode=args.mode, n_trials=args.trials,
        metric=args.metric, fee_bps=args.fee, slippage_bps=args.slippage,
    )

    print(f"\n{result['strategy']} · {result['mode']} · {result['n_trials']} trials "
          f"· optimising {result['metric']}")
    print(search.leaderboard(result, top=args.top).to_string(index=False))
    print("\nbest params: " + ", ".join(f"{k}={v:g}" for k, v in result["best_params"].items()))

    if args.report:
        if not pro:
            _require_pro("HTML report")
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.build(result, data.describe(bars), version=__version__, pro=pro), "utf-8")
        print(f"report -> {out}")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        result["trials"].to_csv(args.csv, index=False)
        print(f"trials -> {args.csv}")
    return 0


def cmd_robust(args: argparse.Namespace) -> int:
    if not _pro():
        _require_pro("Walk-forward audit")
    bars = _load(args)
    best = search.run_search(args.strategy, bars, mode=args.mode, n_trials=args.trials,
                             metric=args.metric, fee_bps=args.fee, slippage_bps=args.slippage)
    wf = robust.walk_forward(args.strategy, bars, folds=args.folds,
                             trials_per_fold=args.trials, mode=args.mode,
                             metric=args.metric, embargo=args.embargo)
    st = robust.stability_map(args.strategy, bars, best["best_params"],
                              fee_bps=args.fee, slippage_bps=args.slippage)

    print("\n" + wf["table"][["fold", "is_sharpe", "oos_sharpe", "decay", "params"]].to_string(index=False))
    s = wf["summary"]
    print(f"\n{s['verdict']}")
    print(f"  in-sample Sharpe  {s['is_sharpe_mean']:.2f}")
    print(f"  out-of-sample     {s['oos_sharpe_mean']:.2f} "
          f"(positive in {s['oos_positive_folds']}/{s['folds']} folds)")
    print(f"  degradation       {s['degradation']:.0%}")
    print(f"  worst OOS drawdown {s['worst_oos_dd']:.1%}")
    print(f"\nstability: {st['verdict']} (plateau ratio {st['plateau_ratio']:.2f})")

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.build(best, data.describe(bars), robust_result=wf,
                                    stability=st, version=__version__), "utf-8")
        print(f"report -> {out}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bars = data.synthetic_bars(rows=args.rows)
    bars.to_csv(out / "demo_bars.csv")

    result = search.run_search(FREE_STRATEGY, bars, mode="grid")
    print(search.leaderboard(result, top=5).to_string(index=False))

    pro = _pro()
    wf = None
    st = None
    if pro:
        wf = robust.walk_forward(FREE_STRATEGY, bars, folds=args.folds, trials_per_fold=20)
        st = robust.stability_map(FREE_STRATEGY, bars, result["best_params"])
        print(f"walk-forward: {wf['summary']['verdict']}")

    html = report.build(result, data.describe(bars), robust_result=wf, stability=st,
                        version=__version__, pro=pro, symbol="DEMO")
    (out / "sweeplab-report.html").write_text(html, "utf-8")
    print(f"report -> {out / 'sweeplab-report.html'} ({'Pro' if pro else 'free tier'})")
    return 0


def cmd_license(args: argparse.Namespace) -> int:
    st = licensing.status()
    if st["valid"]:
        print(f"Pro license · {st['email']} · expires {st['expires'] or 'lifetime'}")
        print(f"source: {licensing.license_path()}")
    else:
        print(f"Free tier · {st['info']}")
        print(f"Pro: {UPLOAD_URL}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    serve(host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sweeplab", description="Parameter sweeps that survive out of sample.")
    parser.add_argument("--version", action="version", version=f"sweeplab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("strategies", help="list built-in strategy templates")
    p.set_defaults(func=cmd_strategies)

    def common(p):
        p.add_argument("-i", "--input", help="CSV/TSV/JSON/Parquet bars")
        p.add_argument("-s", "--strategy", default=FREE_STRATEGY)
        p.add_argument("-m", "--mode", default="grid", choices=["grid", "random", "tpe"])
        p.add_argument("-n", "--trials", type=int, default=60)
        p.add_argument("--metric", default="sharpe")
        p.add_argument("--fee", type=float, default=5.0, help="fee in bps per side")
        p.add_argument("--slippage", type=float, default=2.0, help="slippage in bps per side")

    p = sub.add_parser("sweep", help="search the parameter space")
    common(p)
    p.add_argument("--grid", help='custom space, e.g. "fast=5,10,20;slow=60,90"')
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--report", help="write an HTML report (Pro)")
    p.add_argument("--csv", help="write the full trial table to CSV")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("robust", help="walk-forward audit + stability map (Pro)")
    common(p)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--embargo", type=int, default=5)
    p.add_argument("--report", help="write an HTML report")
    p.set_defaults(func=cmd_robust)

    p = sub.add_parser("demo", help="synthetic end-to-end run, no data needed")
    p.add_argument("-o", "--out", default="sweeplab-demo")
    p.add_argument("--rows", type=int, default=1200)
    p.add_argument("--folds", type=int, default=4)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("serve", help="local web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("license", help="show license status")
    p.set_defaults(func=cmd_license)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
