"""Strategy templates: each takes bars + params and returns a 0/1 position series."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def sma_cross(bars: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    close = bars["close"]
    f = close.rolling(int(fast)).mean()
    s = close.rolling(int(slow)).mean()
    if int(fast) >= int(slow):
        return pd.Series(0.0, index=close.index)
    return (f > s).astype(float)


def rsi_reversion(bars: pd.DataFrame, period: int, buy_below: float, sell_above: float) -> pd.Series:
    rsi = _rsi(bars["close"], int(period))
    raw = pd.Series(np.nan, index=rsi.index)
    raw[rsi < buy_below] = 1.0
    raw[rsi > sell_above] = 0.0
    return raw.ffill().fillna(0.0)


def breakout(bars: pd.DataFrame, channel: int, exit_channel: int) -> pd.Series:
    close = bars["close"]
    upper = close.rolling(int(channel)).max().shift(1)
    lower = close.rolling(int(exit_channel)).min().shift(1)
    pos = np.zeros(len(close))
    state = 0.0
    c = close.to_numpy()
    u = upper.to_numpy()
    l = lower.to_numpy()
    for i in range(len(c)):
        if state == 0 and not np.isnan(u[i]) and c[i] > u[i]:
            state = 1.0
        elif state == 1 and not np.isnan(l[i]) and c[i] < l[i]:
            state = 0.0
        pos[i] = state
    return pd.Series(pos, index=close.index)


def momentum(bars: pd.DataFrame, lookback: int, threshold: float) -> pd.Series:
    roc = bars["close"].pct_change(int(lookback))
    return (roc > threshold).astype(float)


STRATEGIES: Dict[str, Dict] = {
    "sma_cross": {
        "fn": sma_cross,
        "label": "SMA cross",
        "space": {"fast": [5, 10, 20, 30], "slow": [40, 60, 90, 120]},
        "int_params": ["fast", "slow"],
        "pro_only": False,
    },
    "rsi_reversion": {
        "fn": rsi_reversion,
        "label": "RSI mean reversion",
        "space": {"period": [7, 14, 21], "buy_below": [20, 25, 30], "sell_above": [60, 70, 80]},
        "int_params": ["period"],
        "pro_only": True,
    },
    "breakout": {
        "fn": breakout,
        "label": "Channel breakout",
        "space": {"channel": [20, 40, 60], "exit_channel": [10, 20, 30]},
        "int_params": ["channel", "exit_channel"],
        "pro_only": True,
    },
    "momentum": {
        "fn": momentum,
        "label": "Time-series momentum",
        "space": {"lookback": [10, 20, 40, 60], "threshold": [-0.05, 0.0, 0.02, 0.05]},
        "int_params": ["lookback"],
        "pro_only": True,
    },
}


def list_strategies() -> List[Dict]:
    return [
        {"name": k, "label": v["label"], "space": v["space"], "pro_only": v["pro_only"]}
        for k, v in STRATEGIES.items()
    ]


def build_positions(name: str, bars: pd.DataFrame, params: Dict[str, float]) -> pd.Series:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    spec = STRATEGIES[name]
    clean = {}
    for key, value in params.items():
        clean[key] = int(value) if key in spec["int_params"] else float(value)
    pos = spec["fn"](bars, **clean)
    return pos.fillna(0.0).clip(0, 1)


def grid_points(space: Dict[str, List]) -> int:
    total = 1
    for values in space.values():
        total *= len(values)
    return total
