"""02_player_metrics.py: routes -> one row per player.

Reads data/processed/routes.parquet and route_curves.parquet, writes:
  outputs/tables/player_metrics.csv
  outputs/tables/player_speed_curves.csv
Features come from weeks 1-4, the target (avg_separation) from weeks 5-8.
Usage (from repo root):  python src/02_player_metrics.py
"""
import os

import pandas as pd

POSITIONS = ["WR"]   # agreed with Leo: 106 WRs qualify, so WRs only (no position one-hot)
MIN_ROUTES = 20      # per week half
GAS = ["speed_1s", "peak_accel_1s"]
BRAKES = ["peak_decel", "speed_drop"]
FEATURES = GAS + BRAKES
OUT = "outputs/tables"

COLUMNS = ["nfl_id", "player_name", "position", "n_routes_w1_4", "n_routes_w5_8", "top_speed",
           "speed_1s", "peak_accel_1s", "peak_decel", "speed_drop", "share_break_routes", "sep_w1_4",
           "avg_separation", "avg_sep_man_w5_8", "avg_sep_zone_w5_8"]


def main():
    routes = pd.read_parquet("data/processed/routes.parquet")
    routes = routes[routes["position"].isin(POSITIONS)]
    early = routes[routes["week"] <= 4]
    late = routes[routes["week"] >= 5]

    ge = early.groupby("nfl_id")
    gl = late.groupby("nfl_id")
    m = ge[["player_name", "position"]].first()
    m["n_routes_w1_4"] = ge.size()
    m["n_routes_w5_8"] = gl.size()
    m["top_speed"] = ge["route_top_speed"].quantile(0.9)
    # Gas: median (every route starts with a burst). Brakes: 90th percentile, like top_speed,
    # because 69% of windows are cut at the throw, so a median measures how often a WR breaks, not how hard.
    m[GAS] = ge[GAS].median()
    m[BRAKES] = ge[BRAKES].quantile(0.9)
    # share of routes (with a brakes value) that drop more than 2 yd/s
    m["share_break_routes"] = ge["speed_drop"].apply(lambda s: (s.dropna() > 2).mean())
    m["sep_w1_4"] = ge["separation"].mean()
    m["avg_separation"] = gl["separation"].mean()
    m["avg_sep_man_w5_8"] = late[late["coverage_type"] == "Man"].groupby("nfl_id")["separation"].mean()
    m["avg_sep_zone_w5_8"] = late[late["coverage_type"] == "Zone"].groupby("nfl_id")["separation"].mean()

    m = m[(m["n_routes_w1_4"] >= MIN_ROUTES) & (m["n_routes_w5_8"] >= MIN_ROUTES)]
    m["n_routes_w5_8"] = m["n_routes_w5_8"].astype(int)
    m = m.reset_index()[COLUMNS].sort_values("avg_separation", ascending=False)

    curves = pd.read_parquet("data/processed/route_curves.parquet")
    curves = curves[(curves["week"] <= 4) & curves["nfl_id"].isin(m["nfl_id"])]
    sc = curves.groupby(["nfl_id", "t"], as_index=False)["s_smooth"].mean().rename(columns={"s_smooth": "mean_speed"})
    sc = sc.merge(m[["nfl_id", "player_name"]], on="nfl_id")[["nfl_id", "player_name", "t", "mean_speed"]]

    os.makedirs(OUT, exist_ok=True)
    m.to_csv(f"{OUT}/player_metrics.csv", index=False)
    sc.to_csv(f"{OUT}/player_speed_curves.csv", index=False)

    print(f"Players: {len(m)} ({', '.join(POSITIONS)}, {MIN_ROUTES}+ routes in weeks 1-4 and 5-8)")
    print(f"player_speed_curves.csv: {len(sc):,} rows")
    print("\nSpearman correlation (weeks 1-4 features):")
    print(f"{'feature':<16}{'vs top_speed':>14}{'vs avg_separation':>20}")
    for f in ["top_speed"] + FEATURES:
        a = m[f].corr(m["top_speed"], method="spearman")
        b = m[f].corr(m["avg_separation"], method="spearman")
        print(f"{f:<16}{a:>14.2f}{b:>20.2f}")


if __name__ == "__main__":
    main()
