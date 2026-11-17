# Gumroad listing — SweepLab Pro

## Product name
SweepLab Pro — the parameter sweep that tells you the truth

## Price
$34 (launch $24 for the first 100 copies)

## Short description (subtitle)
Walk-forward audit + stability maps for trading strategy parameters. One-time purchase, offline license, no subscription.

## Full description

**Every backtest optimiser hands you the best parameters. None of them tell you the winner is a mirage.**

SweepLab runs your sweep, then does what almost nobody does:

1. **Walk-forward audit** — optimise on a train slice, trade the winning parameters on the next untouched slice, repeat across folds. You see in-sample Sharpe next to out-of-sample Sharpe, fold by fold, with an embargo gap so nothing leaks.
2. **Stability map** — the metric across a 2-D slice of the parameter grid around the winner. A plateau means the edge is real. A spike means you fitted noise.
3. **Honest verdict** — *Holds up out of sample* / *Partial decay* / *Curve-fit*. Most sweeps come back curve-fit. That is the product working.

What you get:

- CLI: `sweep` (grid, random, TPE), `robust` (walk-forward + stability), `demo`, `serve`
- TPE search via Optuna — finds good regions without evaluating every cell
- 4 strategy templates (SMA cross, RSI mean reversion, channel breakout, time-series momentum), custom search spaces
- Costs charged on every position change (fee + slippage in bps), min-trade penalty so "never trade" can't win
- Single-file HTML report: KPIs, equity + drawdown, IS/OOS bars, stability heatmap, leaderboard, verdict banner
- Full trial table export to CSV
- Local drag-and-drop web UI
- Offline Ed25519 license — no phone home, works on an air-gapped machine, lifetime for your email

Free tier: grid sweep on SMA cross from the CLI. Everything above needs Pro.

**Built on Optuna (MIT, ~15k stars) plus an original vectorised backtest core — no look-ahead, positions applied from the next bar.**

## Who it's for
Retail and semi-pro algo traders who already have a strategy and a parameter grid, and who have been burned by a beautiful backtest that died live. Python, pandas-level comfort required.

## What it is not
- Not a broker, not a signal service, not financial advice.
- Not a full backtesting framework (no order book simulation, no portfolio-level multi-asset accounting).

## Files
- `sweeplab-0.1.0-py3-none-any.whl`
- `quickstart.md`
- `sample-audit-report.html`

## Refund
14 days, no questions.

## Tags
backtesting, parameter optimization, walk-forward, overfitting, quantitative trading, python, optuna, sharpe ratio, algo trading
