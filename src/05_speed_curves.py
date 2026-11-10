"""05_speed_curves.py: key chart - same top speed, different brakes.

Reads  outputs/tables/player_metrics.csv, outputs/tables/player_speed_curves.csv
Writes outputs/figures/speed_curves.png

For each pair of WRs with nearly the same top_speed (within 0.2 yd/s) but very
different peak_decel, plots mean speed vs. time after the snap (weeks 1-4 routes)
and shades the braking dip of the hard-braking player.
Usage (from repo root):  python src/05_speed_curves.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

TABLES = "outputs/tables"
OUT = "outputs/figures/speed_curves.png"

# Hand-picked pairs (hard braker, light braker): same top speed, big peak_decel gap,
# names a fan knows. If a name is missing, the script finds the best pair instead.
PAIRS = [
    ("Darius Slayton", "Cooper Kupp"),
    ("Jamison Crowder", "Elijah Moore"),
    ("Sterling Shepard", "Ray-Ray McCloud"),
]
MAX_TOP_SPEED_GAP = 0.2

HARD, LIGHT = "#eb6834", "#2a78d6"   # orange = hard braker, blue = light braker
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"


def find_pairs(m, n=3):
    """Same position, top_speed within 0.2 yd/s, largest peak_decel gap; no player reused."""
    rows = []
    for _, a in m.iterrows():
        for _, b in m.iterrows():
            if (a.position == b.position and a.peak_decel > b.peak_decel
                    and abs(a.top_speed - b.top_speed) <= MAX_TOP_SPEED_GAP):
                rows.append((a.peak_decel - b.peak_decel, a.player_name, b.player_name))
    used, pairs = set(), []
    for _, hard, light in sorted(rows, reverse=True):
        if hard not in used and light not in used:
            pairs.append((hard, light))
            used |= {hard, light}
        if len(pairs) == n:
            break
    return pairs


def dip(curve):
    """Start and end time of the largest fall from a running peak."""
    s = curve["mean_speed"].to_numpy()
    t = curve["t"].to_numpy()
    drop = pd.Series(s).cummax() - s
    end = int(drop.idxmax())
    start = int(pd.Series(s[: end + 1]).idxmax())
    return t[start], t[end], s[start] - s[end]


def main():
    m = pd.read_csv(f"{TABLES}/player_metrics.csv")
    c = pd.read_csv(f"{TABLES}/player_speed_curves.csv").sort_values(["nfl_id", "t"])
    by_name = m.set_index("player_name")

    pairs = [p for p in PAIRS if p[0] in by_name.index and p[1] in by_name.index]
    if len(pairs) < len(PAIRS):
        print("Some hand-picked names missing; finding pairs automatically.")
        pairs = find_pairs(m)

    fig, axes = plt.subplots(1, len(pairs), figsize=(10, 5), sharey=True)
    ymax = c["mean_speed"].max() * 1.12

    for ax, (hard, light) in zip(axes, pairs):
        for name, color, style in [(hard, HARD, "-"), (light, LIGHT, "--")]:
            p = by_name.loc[name]
            curve = c[c["nfl_id"] == p["nfl_id"]]
            ax.plot(curve["t"], curve["mean_speed"], color=color, lw=2, ls=style,
                    label=f"{name}\ntop {p.top_speed:.1f} yd/s · brake {p.peak_decel:.1f} yd/s²"
                          f"\nweeks 5–8 separation {p.avg_separation:.1f} yd")
            if name == hard:
                t0, t1, size = dip(curve)
                if t1 > t0:
                    ax.axvspan(t0, t1, color=HARD, alpha=0.12, lw=0)
                    ax.text((t0 + t1) / 2, ymax * 0.97, f"dip −{size:.1f} yd/s", color=INK,
                            ha="center", va="top", fontsize=9)
            print(f"  {name}: top_speed {p.top_speed:.2f}, peak_decel {p.peak_decel:.2f}, "
                  f"avg_separation w5-8 {p.avg_separation:.2f}")
        ax.set_xlim(0, 3)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Seconds after snap", color=INK)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)
        ax.legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK, handlelength=2.2)

    axes[0].set_ylabel("Mean speed, weeks 1–4 routes (yd/s)", color=INK)
    fig.suptitle("Same top speed, different brakes: hard brakers visibly slow at the break,\n"
                 "but braking didn't predict who got open in weeks 5–8", fontsize=13, color=INK, x=0.02,
                 ha="left")
    fig.text(0.02, 0.01, "Brake = 90th-percentile peak deceleration in the first 3 s. Shaded: the hard braker's "
             "biggest speed dip. 2021 NFL tracking, weeks 1–4.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"Saved {OUT} with {len(pairs)} pairs")


if __name__ == "__main__":
    main()
