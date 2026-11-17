"""SweepLab core tests — strategies, engine, search, robustness, packaging."""

from __future__ import annotations

import base64
import json

import numpy as np
import pandas as pd
import pytest

from sweeplab import data, engine, licensing, report, robust, search
from sweeplab.strategies import STRATEGIES, build_positions


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return data.synthetic_bars(rows=900, seed=3)


def test_positions_are_binary_for_every_strategy(bars):
    for name, spec in STRATEGIES.items():
        params = {k: (v[len(v) // 2] if isinstance(v, list) else v) for k, v in spec["space"].items()}
        pos = build_positions(name, bars, params)
        assert pos.index.equals(bars.index)
        assert set(np.unique(pos.to_numpy())) <= {0.0, 1.0}, name
        assert pos.isna().sum() == 0


def test_sma_cross_rejects_inverted_params(bars):
    pos = build_positions("sma_cross", bars, {"fast": 90, "slow": 10})
    assert pos.sum() == 0


def test_periods_per_year_on_daily_tz_aware_index(bars):
    ppy = engine.periods_per_year(bars.index)
    assert 240 <= ppy <= 370


def test_no_lookahead_first_bar_never_trades(bars):
    pos = pd.Series(1.0, index=bars.index)
    res = engine.backtest(bars, pos)
    assert res["_net"].iloc[0] == 0.0
    # a position held from bar 0 can only earn from bar 1 onward, minus entry cost
    manual = bars["close"].pct_change().fillna(0.0).iloc[1] * 1.0 - (5.0 + 2.0) / 10_000
    assert abs(res["_net"].iloc[1] - manual) < 1e-9


def test_costs_reduce_returns(bars):
    pos = build_positions("sma_cross", bars, {"fast": 5, "slow": 40})
    free = engine.backtest(bars, pos, fee_bps=0.0, slippage_bps=0.0)
    costly = engine.backtest(bars, pos, fee_bps=40.0, slippage_bps=10.0)
    assert free["total_return"] > costly["total_return"]
    assert costly["trades"] == free["trades"]


def test_flat_strategy_has_no_trades_and_is_penalised(bars):
    res = engine.evaluate("sma_cross", bars, {"fast": 90, "slow": 10})
    assert res["trades"] == 0
    assert res["_penalised"] is True
    assert res["total_return"] == pytest.approx(0.0, abs=1e-12)


def test_grid_search_covers_the_whole_space(bars):
    result = search.run_search("sma_cross", bars, mode="grid")
    assert result["n_trials"] == 16
    assert set(result["best_params"]) == set(STRATEGIES["sma_cross"]["space"])
    assert result["best_params"]["fast"] in STRATEGIES["sma_cross"]["space"]["fast"]
    assert result["best_params"]["slow"] in STRATEGIES["sma_cross"]["space"]["slow"]
    assert result["trials"]["score"].is_monotonic_decreasing


def test_tpe_search_beats_a_random_pick(bars):
    result = search.run_search("rsi_reversion", bars, mode="tpe", n_trials=25, seed=5)
    assert len(result["trials"]) >= 10
    median = result["trials"]["score"].median()
    assert result["trials"]["score"].iloc[0] >= median
    assert 5 <= result["best_params"]["period"] <= 21


def test_leaderboard_keeps_parameter_columns(bars):
    result = search.run_search("sma_cross", bars, mode="grid")
    board = search.leaderboard(result, top=3)
    assert list(board.columns[:2]) == ["fast", "slow"]
    assert len(board) == 3


def test_walk_forward_reports_each_fold_once(bars):
    wf = robust.walk_forward("sma_cross", bars, folds=3, trials_per_fold=8)
    table, summary = wf["table"], wf["summary"]
    assert len(table) == 3
    assert table["fold"].tolist() == [1, 2, 3]
    assert summary["verdict"] in {
        "Holds up out of sample",
        "Partial decay — tradeable with caution",
        "Curve-fit — the winner does not survive",
    }
    assert 0 <= summary["oos_positive_ratio"] <= 1
    assert "degradation" in summary


def test_walk_forward_rejects_short_data():
    with pytest.raises(ValueError):
        robust.walk_forward("sma_cross", data.synthetic_bars(rows=60), folds=4)


def test_stability_map_finds_a_plateau_or_a_spike(bars):
    result = search.run_search("sma_cross", bars, mode="grid")
    st = robust.stability_map("sma_cross", bars, result["best_params"])
    assert st["matrix"].shape == (len(st["y_values"]), len(st["x_values"]))
    assert st["best"]["value"] == pytest.approx(st["matrix"].max())
    assert 0.0 <= st["plateau_ratio"] <= 2.0
    assert st["verdict"] in {"Plateau — edge is robust", "Smooth slope — acceptable",
                             "Spike — single lucky point"}


def test_bad_license_falls_back_to_free(tmp_path, monkeypatch):
    doc = {"payload": {"product": "sweeplab", "email": "x@y.z", "expires": "2030-01-01"},
           "sig": base64.b64encode(b"not-a-signature").decode()}
    path = tmp_path / "license.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setenv("SWEEPLAB_LICENSE", str(path))
    st = licensing.status()
    assert st["valid"] is False and st["tier"] == "free"


def test_license_for_another_product_is_rejected(tmp_path, monkeypatch):
    doc = {"payload": {"product": "alphaledger", "email": "x@y.z"}, "sig": "AAAA"}
    path = tmp_path / "license.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setenv("SWEEPLAB_LICENSE", str(path))
    assert licensing.status()["valid"] is False


def test_report_embeds_images_as_data_uris(bars):
    result = search.run_search("sma_cross", bars, mode="grid")
    html = report.build(result, data.describe(bars), pro=False)
    assert html.startswith("<!doctype html>")
    assert '<img src="data:image/png;base64,' in html
    assert "Walk-forward audit is a Pro feature" in html
    assert "iVBORw0KGgo" in html and "iVBORw0KGgo\n" not in html


def test_report_pro_sections_have_no_lock(bars):
    result = search.run_search("sma_cross", bars, mode="grid")
    wf = robust.walk_forward("sma_cross", bars, folds=2, trials_per_fold=6)
    st = robust.stability_map("sma_cross", bars, result["best_params"])
    html = report.build(result, data.describe(bars), robust_result=wf, stability=st, pro=True)
    assert "Pro feature" not in html
    assert "Out-of-sample survival" in html


def test_data_loader_accepts_loose_columns(tmp_path):
    frame = pd.DataFrame({
        "Time": pd.date_range("2024-01-01", periods=50, freq="1D"),
        "Close": np.linspace(100, 120, 50),
        "High": np.linspace(101, 121, 50),
        "Low": np.linspace(99, 119, 50),
        "Volume": np.arange(50),
    })
    path = tmp_path / "bars.csv"
    frame.to_csv(path, index=False)
    loaded = data.load_bars(path)
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert len(loaded) == 50


def test_missing_price_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2, 3]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        data.load_bars(path)
