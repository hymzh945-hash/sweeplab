# SweepLab

**Parameter sweeps that survive out of sample.**

Every backtesting tool gives you "the best parameters". Almost none tell you whether those
parameters will still work next month. SweepLab runs the sweep, then puts the winner through a
purged walk-forward audit and a neighbourhood stability map, so you can tell a real edge from a
lucky grid cell.

Built on [Optuna](https://github.com/optuna/optuna) (MIT) for the search and an original
vectorised backtest core (no look-ahead, costs charged on every position change).

```
$ sweeplab sweep -i btc_1d.csv --strategy sma_cross --mode tpe -n 120
$ sweeplab robust -i btc_1d.csv --strategy sma_cross --folds 4 --report audit.html
```

## Why another optimiser

| What most tools do | What SweepLab does |
|---|---|
| Rank 5,000 combinations and hand you the top row | Rank them, then test the top row out of sample |
| Report Sharpe on the whole history | Re-optimise per fold, trade the winner on the next slice |
| Ignore that the winner sits between two losers | Score the 8 cells around it — plateau or spike |
| Charge nothing for turnover | Fee + slippage in basis points on every position change |

The honest answer is often "this doesn't survive" — and that is the whole point.

## Install

```bash
pip install sweeplab                 # free tier
pip install "sweeplab[license]"      # + offline license verification
```

## Free vs Pro

| | Free | Pro |
|---|---|---|
| `sma_cross` grid sweep | ✅ | ✅ |
| TPE / random search | — | ✅ |
| `rsi_reversion`, `breakout`, `momentum` | — | ✅ |
| Walk-forward out-of-sample audit | — | ✅ |
| Stability (plateau vs spike) map | — | ✅ |
| HTML report + trial CSV export | — | ✅ |
| Local web UI | grid only | full |

Pro is a one-time purchase, activated offline with a signed key — no phone-home, no seat server.
Get it at <https://sweeplab.gumroad.com>.

## Commands

```bash
sweeplab strategies                       # what's built in and their search spaces
sweeplab demo                             # synthetic end-to-end run, no data needed
sweeplab sweep  -i bars.csv -s sma_cross -m tpe -n 120 --csv trials.csv
sweeplab robust -i bars.csv -s sma_cross --folds 4 --report audit.html
sweeplab serve                            # drag-and-drop UI on :8765
sweeplab license                          # show tier
```

Custom search space:

```bash
sweeplab sweep -i bars.csv --grid "fast=3,5,8,13;slow=30,55,90,120,200"
```

## Input data

Any CSV/TSV/JSON/Parquet with a close column (`close`, `Close`, `price`, `last`) and optionally
`open/high/low/volume`. A `date`/`time`/`timestamp` column becomes the index.

### Choosing the metric

`--metric sharpe` (default), `total_return`, `cagr`, `calmar`, `sortino`, `profit_factor`. Any
strategy with fewer than 3 round trips is penalised, so the sweep can't win by never trading.

## What the audit tells you

- **Walk-forward**: expanding-window folds with an embargo gap. Each fold optimises on the train
  slice and trades the winner on the next, untouched slice. Verdicts: *holds up*, *partial decay*,
  *curve-fit*.
- **Stability map**: the metric across a 2-D slice of the parameter grid around the winner.
  Plateau ratio ≥ 0.8 = robust, < 0.55 = spike.

## Licence & attribution

SweepLab's code is MIT. Optuna is MIT and used as a dependency (see `NOTICE`). Not affiliated with
Optuna or Preferred Networks. Nothing here is financial advice; past performance is not indicative
of future results.

## Pro edition

The paid edition adds the HTML report templates, CSV exports, and an offline-verified signed
licence — no server check, no telemetry. $34 once, yours to keep:

https://hymz.gumroad.com/l/sweeplab

