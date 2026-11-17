"""Does the winning parameter set survive out of sample, or is it a spike?"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import engine, search
from .strategies import STRATEGIES


def _slice(bars: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return bars.iloc[start:end]


def walk_forward(
    strategy: str,
    bars: pd.DataFrame,
    folds: int = 4,
    trials_per_fold: int = 40,
    mode: str = "tpe",
    metric: str = "sharpe",
    embargo: int = 5,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> Dict:
    """Optimise on each train slice, then measure the winner on the next slice.

    `embargo` bars are dropped between train and test so indicator warm-up and
    the signal of the last train bar cannot leak into the test window.
    """
    n = len(bars)
    if n < folds * 30:
        raise ValueError("not enough bars for the requested number of folds")

    # Expanding window: the first segment is pure training data, then each
    # subsequent segment is an out-of-sample fold.
    bounds = np.linspace(0, n, folds + 2).astype(int)
    rows: List[Dict] = []

    for i in range(1, folds + 1):
        test_start, test_end = int(bounds[i]), int(bounds[i + 1])
        train_end = max(60, test_start - embargo)
        train = _slice(bars, 0, train_end)
        test = _slice(bars, test_start, test_end)
        if len(train) < 60 or len(test) < 20:
            continue

        res = search.run_search(strategy, train, mode=mode, n_trials=trials_per_fold,
                                metric=metric, fee_bps=fee_bps, slippage_bps=slippage_bps, seed=7 + i)
        params = res["best_params"]
        is_res = engine.evaluate(strategy, train, params, fee_bps, slippage_bps)
        oos_res = engine.evaluate(strategy, test, params, fee_bps, slippage_bps)

        rows.append({
            "fold": len(rows) + 1,
            "train_start": str(train.index[0].date()),
            "train_end": str(train.index[-1].date()),
            "test_start": str(test.index[0].date()),
            "test_end": str(test.index[-1].date()),
            "params": ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()),
            "is_sharpe": round(is_res["sharpe"], 3),
            "oos_sharpe": round(oos_res["sharpe"], 3),
            "is_return": round(is_res["total_return"], 4),
            "oos_return": round(oos_res["total_return"], 4),
            "oos_max_dd": round(oos_res["max_drawdown"], 4),
            "decay": round(_decay(is_res["sharpe"], oos_res["sharpe"]), 3),
        })

    if not rows:
        raise RuntimeError("no fold produced a usable result")

    table = pd.DataFrame(rows)
    oos_pos = int((table["oos_sharpe"] > 0).sum())
    is_mean = float(table["is_sharpe"].mean())
    oos_mean = float(table["oos_sharpe"].mean())
    summary = {
        "folds": len(table),
        "is_sharpe_mean": is_mean,
        "oos_sharpe_mean": oos_mean,
        "oos_positive_folds": oos_pos,
        "oos_positive_ratio": oos_pos / len(table),
        "decay_mean": float(table["decay"].mean()),
        "worst_oos_dd": float(table["oos_max_dd"].min()),
        "degradation": round(_decay(is_mean, oos_mean), 3),
    }
    summary["verdict"], summary["verdict_color"] = _verdict(summary)
    return {"table": table, "summary": summary}


def _decay(is_value: float, oos_value: float) -> float:
    if abs(is_value) < 1e-9:
        return 0.0
    return 1 - (oos_value / is_value)


def _verdict(s: Dict) -> tuple[str, str]:
    if s["oos_positive_ratio"] >= 0.75 and s["oos_sharpe_mean"] >= 0.4 and s["degradation"] <= 0.45:
        return "Holds up out of sample", "#34d399"
    if s["oos_positive_ratio"] >= 0.5 and s["oos_sharpe_mean"] >= 0.15:
        return "Partial decay — tradeable with caution", "#fbbf24"
    return "Curve-fit — the winner does not survive", "#f87171"


def stability_map(
    strategy: str,
    bars: pd.DataFrame,
    best_params: Dict,
    space: Optional[Dict[str, List]] = None,
    axes: Optional[List[str]] = None,
    metric: str = "sharpe",
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> Dict:
    """Score the neighbourhood of the winning point.

    A real edge sits on a plateau: neighbouring parameters perform similarly.
    A spike means the win came from one lucky combination.
    """
    space = space or STRATEGIES[strategy]["space"]
    keys = axes or list(space.keys())[:2]
    if len(keys) < 2:
        keys = list(space.keys())[:2] + list(space.keys())[:1]
    x_key, y_key = keys[0], keys[1]

    xs = space[x_key]
    ys = space[y_key]
    matrix = np.zeros((len(ys), len(xs)))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            params = dict(best_params)
            params[x_key] = x
            params[y_key] = y
            res = engine.evaluate(strategy, bars, params, fee_bps, slippage_bps)
            value = engine.score(res, metric)
            matrix[j, i] = value

    best_index = np.unravel_index(np.argmax(matrix), matrix.shape)
    best_value = float(matrix[best_index])
    neighbours = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = best_index[0] + dy, best_index[1] + dx
            if 0 <= ny < matrix.shape[0] and 0 <= nx < matrix.shape[1]:
                neighbours.append(float(matrix[ny, nx]))

    mean_neighbours = float(np.mean(neighbours)) if neighbours else best_value
    ratio = mean_neighbours / best_value if best_value > 0 else 0.0
    verdict = ("Plateau — edge is robust" if ratio >= 0.8
               else "Smooth slope — acceptable" if ratio >= 0.55
               else "Spike — single lucky point")

    return {
        "axes": [x_key, y_key],
        "x_values": xs,
        "y_values": ys,
        "matrix": matrix,
        "best": {"x": xs[best_index[1]], "y": ys[best_index[0]], "value": best_value},
        "neighbour_mean": mean_neighbours,
        "plateau_ratio": round(ratio, 3),
        "verdict": verdict,
    }
