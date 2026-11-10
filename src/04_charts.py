"""04_charts.py: model comparison + winning feature vs. future separation.

Reads outputs/tables/*.csv, writes:
  outputs/figures/model_comparison.png
  outputs/figures/metric_vs_separation.png
Usage (from repo root):  python src/04_charts.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES = "outputs/tables"
FIGS = "outputs/figures"
FEATURE = "speed_drop"   # strongest gas/brakes feature (Spearman with avg_separation); confirm with Leo
FEATURE_LABEL = {"speed_drop": "Biggest speed drop in first 3 s, 90th pct route (yd/s)",
                 "peak_decel": "Hardest deceleration in first 3 s, 90th pct route (yd/s²)"}

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def model_comparison():
    res = pd.read_csv(f"{TABLES}/model_results.csv")
    wide = res.pivot_table(index=["repeat", "fold"], columns="feature_set", values="mae_yards")
    mean = wide.mean()
    improve = (mean["baseline"] - mean["full"]) / mean["baseline"] * 100
    wins = int((wide["full"] < wide["baseline"]).sum())
    verdict = "Gas & brakes help" if improve > 0 else "Top speed is enough"

    fig, ax = plt.subplots(figsize=(8, 5))
    # one thin gray line per fold joins the two models' errors on the same split
    for _, r in wide.iterrows():
        ax.plot([0, 1], [r["baseline"], r["full"]], color=GRID, lw=1, zorder=1)
    rng = np.random.default_rng(0)
    for i, (name, color) in enumerate([("baseline", BLUE), ("full", ORANGE)]):
        x = i + rng.uniform(-0.04, 0.04, len(wide))
        ax.scatter(x, wide[name], s=36, color=color, alpha=0.55, edgecolor=SURFACE, lw=1, zorder=2)
        sd = wide[name].std()
        ax.errorbar(i + 0.18, mean[name], yerr=sd, fmt="o", ms=10, color=color, ecolor=color,
                    elinewidth=2, capsize=6, zorder=3)
        ax.annotate(f"{mean[name]:.3f} yd", (i + 0.24, mean[name]), va="center", color=INK, fontsize=12)
    ax.set_xticks([0, 1], ["Baseline\n(top speed only)", "Full\n(top speed + gas + brakes)"])
    ax.set_xlim(-0.4, 1.6)
    ax.set_ylabel("Mean absolute error, weeks 5–8 separation (yards)")
    change = f"cuts error {improve:.1f}%" if improve > 0 else f"makes error {-improve:.1f}% worse"
    ax.set_title(f"{verdict}: adding gas & brakes {change}\n"
                 f"(full model better in {wins} of {len(wide)} folds; 106 WRs, 5×5 cross-validation)")
    ax.text(1.58, ax.get_ylim()[0], "dots = folds · bar = mean ± 1 SD", ha="right", va="bottom",
            color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/model_comparison.png", dpi=150)
    plt.close(fig)
    return improve, wins


def metric_vs_separation():
    m = pd.read_csv(f"{TABLES}/player_metrics.csv")
    x, y = m[FEATURE], m["avg_separation"]
    rho = x.corr(y, method="spearman")
    slope, icpt = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=48, color=BLUE, alpha=0.75, edgecolor=SURFACE, lw=1.5, zorder=2)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, icpt + slope * xs, color=INK2, lw=2, ls="--", zorder=1, label="Linear trend")

    # standouts: most / least open, hardest / softest braker
    picks = [y.idxmax(), y.idxmin(), x.idxmax(), x.idxmin()]
    for i in dict.fromkeys(picks):
        ax.annotate(m.loc[i, "player_name"], (x[i], y[i]), xytext=(6, 6), textcoords="offset points",
                    fontsize=11, color=INK)
        ax.scatter(x[i], y[i], s=48, facecolor="none", edgecolor=INK, lw=1.5, zorder=3)

    ax.set_xlabel(f"Weeks 1–4: {FEATURE_LABEL.get(FEATURE, FEATURE)}")
    ax.set_ylabel("Weeks 5–8: avg separation (yards)")
    ax.set_title(f"Harder brakers get only slightly more open later (Spearman {rho:.2f})\n"
                 f"{len(m)} WRs; each dot is one receiver")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/metric_vs_separation.png", dpi=150)
    plt.close(fig)
    return rho


def main():
    os.makedirs(FIGS, exist_ok=True)
    improve, wins = model_comparison()
    rho = metric_vs_separation()
    print(f"model_comparison.png: improvement {improve:+.1f}%, full wins {wins}/25 folds")
    print(f"metric_vs_separation.png: {FEATURE} vs avg_separation, Spearman {rho:.2f}")


if __name__ == "__main__":
    main()
