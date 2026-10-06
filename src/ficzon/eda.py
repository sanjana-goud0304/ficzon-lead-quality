"""Reusable EDA helpers: Wilson intervals, level grouping, and chart builders.

Design choices (Phase 3):
  * Every rate is shown with its sample size and a 95% Wilson interval, so small
    cells (e.g. 51 "Existing Client" rows) are not read as firm facts.
  * One accent hue (blue) carries "High Potential"; neutral gray carries baselines
    and "Low" / context. Magnitude uses a single-hue sequential ramp.
  * Two measures on different scales are separate panels - never a dual axis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Palette (reference instance of the dataviz skill; light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
BLUE_SOFT = "#9cbfec"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#8f8e88"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eaf2fc", "#2a78d6", "#123c75"])

MISSING = "(missing)"
OTHER = "Other (<{n} rows each)"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "figure.dpi": 110, "savefig.dpi": 150,
    })


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def group_levels(s: pd.Series, min_n: int = 30) -> pd.Series:
    """Fill missing with an explicit label and fold levels under min_n into one Other level."""
    s = s.astype("string").fillna(MISSING)
    counts = s.value_counts()
    keep = set(counts[counts >= min_n].index) | {MISSING}
    return s.where(s.isin(keep), OTHER.format(n=min_n))


def rate_table(df: pd.DataFrame, by: str | pd.Series, target: str = "target_high") -> pd.DataFrame:
    g = df.groupby(by, observed=True)[target].agg(n="size", k="sum")
    g["rate"] = g.k / g.n
    g["lo"], g["hi"] = wilson(g.k, g.n)
    return g.reset_index()


def barh_rate(ax, tbl: pd.DataFrame, label_col: str, overall: float, title: str, subtitle: str = "") -> None:
    """Horizontal bars of High-Potential rate with 95% CI, n labels, and an overall baseline."""
    t = tbl.copy()
    other_mask = t[label_col].astype(str).str.startswith("Other (")
    t = pd.concat([t[other_mask], t[~other_mask].sort_values("rate")])  # 'Other' at the bottom
    y = np.arange(len(t))
    colors = [BLUE if n >= 100 else BLUE_SOFT for n in t.n]
    ax.barh(y, t.rate * 100, color=colors, height=0.62, zorder=2)
    ax.errorbar(t.rate * 100, y, xerr=[(t.rate - t.lo) * 100, (t.hi - t.rate) * 100],
                fmt="none", ecolor=INK_2, elinewidth=1, capsize=2.5, zorder=3)
    ax.axvline(overall * 100, color=GRAY, ls="--", lw=1.2, zorder=1)
    for yi, (r, hi, n) in enumerate(zip(t.rate, t.hi, t.n)):
        ax.text(min(hi * 100 + 1.5, 101), yi, f"{r*100:.0f}%  (n={n:,})", va="center", fontsize=8.5, color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8, alpha=0.9), zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(t[label_col])
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("High Potential rate (%)  -  whiskers: 95% interval")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, pad=24 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    ax.legend(handles=[Patch(color=BLUE, label="n >= 100"), Patch(color=BLUE_SOFT, label="n < 100 (wide uncertainty)"),
                       plt.Line2D([], [], color=GRAY, ls="--", label=f"Overall {overall*100:.1f}%")],
              loc="lower right", fontsize=8, frameon=False)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013) between two categorical series."""
    ct = pd.crosstab(x, y).to_numpy().astype(float)
    n = ct.sum()
    expected = ct.sum(1, keepdims=True) @ ct.sum(0, keepdims=True) / n
    chi2 = ((ct - expected) ** 2 / expected).sum()
    r, k = ct.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(kc - 1, rc - 1))))
