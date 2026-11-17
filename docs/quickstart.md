# SweepLab Pro — quick start

1. `pip install "sweeplab[license,parquet]"`
2. Save your key to `~/sweeplab-license.json` (or `export SWEEPLAB_LICENSE=/path/to/key.json`)
3. `sweeplab license` → should print your email and expiry
4. `sweeplab demo` → synthetic run with the full audit, confirms everything works
5. On your own data:
   `sweeplab sweep -i bars.csv -s sma_cross -m tpe -n 120 --csv trials.csv`
   `sweeplab robust -i bars.csv -s sma_cross --folds 4 --report audit.html`
6. `sweeplab serve` → drag-and-drop UI at http://127.0.0.1:8765

## Reading the output

- **IS vs OOS Sharpe** — if OOS is less than half of IS, you fitted the past.
- **Verdict** — `Holds up out of sample` needs ≥75% positive folds and mean OOS Sharpe ≥ 0.4.
- **Plateau ratio** — neighbours within ±1 grid step divided by the best cell. ≥0.8 is a plateau.

## Practical notes

- Use enough history: 3+ years daily, or 2+ years hourly, for a 4-fold audit.
- Costs matter: `--fee 5 --slippage 2` is optimistic for retail crypto; try 10/5 too.
- A negative verdict is information, not a failure — it saved you a live drawdown.

Questions: reply to your Gumroad receipt.
