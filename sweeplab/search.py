"""Parameter search on top of Optuna (MIT, ~15k stars).

Grid, random and TPE search share one interface, and every trial keeps its full
metric set so the leaderboard can be inspected afterwards — not just the winner.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import numpy as np
import optuna
import pandas as pd

from . import engine
from .strategies import STRATEGIES

optuna.logging.set_verbosity(optuna.logging.WARNING)

METRIC_KEYS = ["sharpe", "total_return", "cagr", "max_drawdown", "sortino",
               "win_rate", "profit_factor", "trades", "exposure", "calmar"]


def _space_for(strategy: str, space: Optional[Dict[str, List]] = None) -> Dict[str, List]:
    if space:
        return space
    return STRATEGIES[strategy]["space"]


def _int_params(strategy: str) -> List[str]:
    return STRATEGIES[strategy]["int_params"]


def _search_space(strategy: str, space: Dict[str, List], mode: str) -> Dict:
    ints = _int_params(strategy)
    out = {}
    for key, values in space.items():
        if mode == "tpe":
            lo, hi = float(min(values)), float(max(values))
            out[key] = optuna.distributions.IntDistribution(int(lo), int(hi)) if key in ints \
                else optuna.distributions.FloatDistribution(lo, hi)
        else:
            out[key] = [int(v) for v in values] if key in ints else [float(v) for v in values]
    return out


def _sampler(mode: str, search_space: Dict, seed: int):
    if mode == "grid":
        return optuna.samplers.GridSampler(search_space, seed=seed)
    if mode == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed, n_startup_trials=12)


def run_search(
    strategy: str,
    bars: pd.DataFrame,
    space: Optional[Dict[str, List]] = None,
    mode: str = "grid",
    n_trials: int = 100,
    metric: str = "sharpe",
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    min_trades: int = 3,
    seed: int = 7,
) -> Dict:
    """Run a sweep and return trials, leaderboard and the best parameter set."""
    if mode not in {"grid", "random", "tpe"}:
        raise ValueError("mode must be grid, random or tpe")
    space = _space_for(strategy, space)
    search_space = _search_space(strategy, space, mode)
    if mode == "grid":
        n_trials = int(np.prod([len(v) for v in space.values()]))

    records: List[Dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {k: trial.suggest_auto(k, v) if hasattr(trial, "suggest_auto")
                  else _suggest(trial, k, v) for k, v in search_space.items()}
        res = engine.evaluate(strategy, bars, params, fee_bps, slippage_bps, min_trades)
        row = {**params, **{k: res.get(k) for k in METRIC_KEYS},
               "score": engine.score(res, metric), "penalised": bool(res.get("_penalised"))}
        records.append(row)
        for k, v in row.items():
            if isinstance(v, (int, float, bool)):
                trial.set_user_attr(k, v)
        return row["score"]

    sampler = _sampler(mode, search_space, seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))

    trials = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)
    if trials.empty:
        raise RuntimeError("no trials completed — check the data and parameters")

    best = trials.iloc[0].to_dict()
    best_params = {k: best[k] for k in space}
    best_res = engine.evaluate(strategy, bars, best_params, fee_bps, slippage_bps, min_trades)

    return {
        "strategy": strategy,
        "mode": mode,
        "metric": metric,
        "space": space,
        "n_trials": len(trials),
        "trials": trials,
        "best_params": best_params,
        "best_metrics": best_res,
        "best_equity": best_res["_equity"],
        "study": study,
    }


def _suggest(trial: optuna.Trial, name: str, dist) -> float:
    if isinstance(dist, list):
        return trial.suggest_categorical(name, dist)
    if isinstance(dist, optuna.distributions.IntDistribution):
        return trial.suggest_int(name, dist.low, dist.high)
    return trial.suggest_float(name, dist.low, dist.high)


def leaderboard(result: Dict, top: int = 10) -> pd.DataFrame:
    cols = list(result["space"].keys()) + ["score", "sharpe", "total_return", "max_drawdown", "trades"]
    return result["trials"][cols].head(top).round(4)
