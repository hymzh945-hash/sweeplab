"""Vectorised backtest core: positions in, net returns and metrics out."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .strategies import build_positions


def _naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """pandas 3 refuses tz-aware -> naive astype, so convert explicitly."""
    if getattr(index, "tz", None) is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


def periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    index = _naive(index)
    deltas = np.diff(index.astype("datetime64[ns]").astype("int64")) / 1e9
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return 252.0
    median = float(np.median(deltas))
    if median < 86400:
        return round(365 * 24 * 3600 / median)
    span = (index[-1] - index[0]).total_seconds() / 86400
    if span <= 0:
        return 252.0
    raw = (len(index) - 1) / (span / 365.25)
    for cand in (365.0, 252.0, 52.0, 12.0, 4.0, 1.0):
        if abs(raw - cand) / cand < 0.15:
            return cand
    return round(raw)


def backtest(
    bars: pd.DataFrame,
    positions: pd.Series,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> Dict:
    """Trade the next bar after a signal, charge costs on every position change."""
    close = bars["close"].astype(float)
    asset_ret = close.pct_change().fillna(0.0)

    pos = positions.reindex(close.index).fillna(0.0).clip(0, 1)
    held = pos.shift(1).fillna(0.0)  # decisions apply from the next bar
    gross = held * asset_ret

    turnover = held.diff().abs().fillna(held.abs())
    cost = turnover * (fee_bps + slippage_bps) / 10_000
    net = gross - cost

    equity = (1 + net).cumprod()
    dd = equity / equity.cummax() - 1

    ppy = periods_per_year(close.index)
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    years = len(net) / ppy
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else total
    vol = float(net.std(ddof=1) * math.sqrt(ppy)) if len(net) > 2 else 0.0
    downside = net[net < 0]
    dvol = float(downside.std(ddof=1) * math.sqrt(ppy)) if len(downside) > 2 else 0.0

    trades = int((turnover > 1e-9).sum() / 2) if len(turnover) else 0
    wins = float(net[net > 0].sum())
    losses = float(abs(net[net < 0].sum()))

    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe": float(net.mean() * ppy / vol) if vol else 0.0,
        "sortino": float(net.mean() * ppy / dvol) if dvol else 0.0,
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "calmar": float(cagr / abs(dd.min())) if len(dd) and dd.min() < 0 else 0.0,
        "volatility": vol,
        "win_rate": float((net > 0).mean()) if len(net) else 0.0,
        "profit_factor": float(wins / losses) if losses else float("inf"),
        "trades": trades,
        "exposure": float(held.mean()) if len(held) else 0.0,
        "final_equity": float(equity.iloc[-1]) if len(equity) else 1.0,
        "_equity": equity,
        "_net": net,
    }


def evaluate(name: str, bars: pd.DataFrame, params: Dict, fee_bps: float = 5.0,
             slippage_bps: float = 2.0, min_trades: int = 3) -> Dict:
    """Positions + backtest in one call, with a trade-count penalty for degenerate params."""
    pos = build_positions(name, bars, params)
    res = backtest(bars, pos, fee_bps=fee_bps, slippage_bps=slippage_bps)
    if res["trades"] < min_trades:
        res["sharpe"] = res["sharpe"] * 0.5
        res["_penalised"] = True
    return res


def score(res: Dict, metric: str = "sharpe") -> float:
    value = res.get(metric, 0.0)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return float(value)
