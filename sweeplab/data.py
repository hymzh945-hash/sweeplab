"""Bar loading: OHLCV from CSV/TSV/JSON/Parquet with loose column detection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TIME_COLUMNS = {"date", "datetime", "time", "timestamp", "open_time", "index"}
CLOSE_ALIASES = {"close", "c", "price", "last", "close_price", "adj close", "adj_close"}
OHLC = {"open": {"open", "o"}, "high": {"high", "h"}, "low": {"low", "l"},
        "close": CLOSE_ALIASES, "volume": {"volume", "vol", "v", "qty"}}


def _find(columns, aliases) -> str | None:
    lowered = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def load_bars(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".json", ".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=suffix != ".json")
    else:
        df = pd.read_csv(path, sep=None, engine="python")

    df.columns = [str(c) for c in df.columns]
    time_col = _find(df.columns, TIME_COLUMNS)
    if time_col and time_col != "index":
        index = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.drop(columns=[time_col])
        df.index = index
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except (ValueError, TypeError):
            df.index = pd.RangeIndex(len(df))

    out = {}
    for name, aliases in OHLC.items():
        col = _find(df.columns, aliases)
        if col is None:
            if name == "close":
                raise ValueError("no close/price column found in the file")
            continue
        out[name] = pd.to_numeric(df[col], errors="coerce")

    bars = pd.DataFrame(out)
    for name in ("open", "high", "low"):
        if name not in bars and "close" in bars:
            bars[name] = bars["close"]
    if "volume" not in bars:
        bars["volume"] = 0.0
    bars = bars[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    bars.index.name = "timestamp"
    return bars


def describe(bars: pd.DataFrame) -> dict:
    return {
        "rows": int(len(bars)),
        "start": str(bars.index[0]) if len(bars) else "",
        "end": str(bars.index[-1]) if len(bars) else "",
        "close_min": float(bars["close"].min()) if len(bars) else 0.0,
        "close_max": float(bars["close"].max()) if len(bars) else 0.0,
    }


def synthetic_bars(rows: int = 1200, seed: int = 11, drift: float = 0.0003) -> pd.DataFrame:
    """Deterministic random walk with regime shifts — no network needed."""
    import numpy as np

    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=rows, freq="1D", tz="UTC")
    vol = np.where((np.arange(rows) // 120) % 2 == 0, 0.012, 0.022)
    rets = rng.normal(drift, vol)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0.002, 0.012, rows))
    low = close * (1 - rng.uniform(0.002, 0.012, rows))
    open_ = np.concatenate([[close[0]], close[:-1]])
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(1000, 50000, rows)}, index=index)
    bars.index.name = "timestamp"
    return bars
