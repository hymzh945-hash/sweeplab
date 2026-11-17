"""Chart helpers — matplotlib Agg to base64 PNG, dark theme."""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG = "#0f1420"
PANEL = "#161d2e"
FG = "#e6edf7"
MUTED = "#8b98ad"
ACCENT = "#38bdf8"
GOOD = "#34d399"
BAD = "#f87171"
WARN = "#fbbf24"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "axes.edgecolor": "#243046",
    "axes.labelcolor": FG,
    "text.color": FG,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 9,
    "axes.grid": True,
    "grid.color": "#1e2739",
    "grid.linewidth": 0.6,
})


def _save(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def equity_curve(equity: pd.Series) -> str:
    """`equity` is already a compounded equity curve (not per-bar returns)."""
    dd = equity / equity.cummax() - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06})
    ax1.plot(equity.index, equity.values, color=ACCENT, lw=1.4)
    ax1.fill_between(equity.index, equity.values, 1, color=ACCENT, alpha=0.10)
    ax1.set_title("Strategy equity (net of costs)", loc="left", color=FG, fontsize=11)
    ax1.set_ylabel("equity")

    ax2.fill_between(dd.index, dd.values, 0, color=BAD, alpha=0.55)
    ax2.set_title("Drawdown", loc="left", color=MUTED, fontsize=9)
    ax2.set_ylabel("dd")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    return _save(fig)


def heatmap(matrix: np.ndarray, x_values, y_values, x_label: str, y_label: str) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.max(finite)) if finite.size else 1.0
    vmin = float(np.min(finite)) if finite.size else 0.0
    im = ax.imshow(matrix, cmap="magma", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([f"{v:g}" for v in x_values])
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([f"{v:g}" for v in y_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    span = (vmax - vmin) or 1.0
    for j in range(matrix.shape[0]):
        for i in range(matrix.shape[1]):
            value = matrix[j, i]
            if not np.isfinite(value):
                continue
            colour = "#111827" if (value - vmin) / span > 0.55 else "#f8fafc"
            ax.text(i, j, f"{value:.2f}", ha="center", va="center", fontsize=8, color=colour)

    ax.set_title("Sharpe by parameter pair", loc="left", color=FG, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02).ax.tick_params(colors=MUTED)
    return _save(fig)


def walkforward_bars(table: pd.DataFrame) -> str:
    folds = table["fold"].astype(str).tolist()
    x = np.arange(len(folds))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.bar(x - width / 2, table["is_sharpe"], width, label="in-sample", color=MUTED)
    ax.bar(x + width / 2, table["oos_sharpe"], width, label="out-of-sample", color=ACCENT)
    ax.axhline(0, color="#243046", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {f}" for f in folds])
    ax.set_ylabel("Sharpe")
    ax.set_title("Same parameters, optimised in-sample and traded out-of-sample", loc="left",
                 color=FG, fontsize=11)
    ax.legend(frameon=False, labelcolor=FG, loc="upper left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig)
