"""01_build_routes.py: tracking -> one row per route, plus per-route speed curves.

Reads data/raw/*, writes:
  data/processed/routes.parquet        (one row per route)
  data/processed/route_curves.parquet  (one row per route per frame, first 3 s)

Keeps WR/TE/RB route runners on every route; 02 picks the position group.
Usage (from repo root):  python src/01_build_routes.py [--n-games 10]
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

RAW = "data/raw"
OUT = "data/processed"
COLS = ["gameId", "playId", "nflId", "frameId", "team", "playDirection", "x", "y", "s", "event"]
THROW_EVENTS = {"pass_forward", "autoevent_passforward"}
KEY = ["playId", "nflId"]

MEASURE_FRAMES = 25   # separation measured at snap + 2.5 s (or the throw if earlier)
WINDOW_FRAMES = 30    # movement window: snap to snap + 3.0 s, cut at the throw
MIN_WINDOW = 15       # shorter windows get no brakes features
ONE_SEC = 10          # frames in 1 s


def load_lookups():
    games = pd.read_csv(f"{RAW}/games.csv", usecols=["gameId", "week"])
    plays = pd.read_csv(f"{RAW}/plays.csv", usecols=["gameId", "playId", "possessionTeam", "defensiveTeam",
                                                    "passResult", "pff_passCoverageType"])
    plays = plays[plays["passResult"].isin(["C", "I", "IN"])].merge(games, on="gameId")
    players = pd.read_csv(f"{RAW}/players.csv", usecols=["nflId", "officialPosition", "displayName"])
    pff = pd.read_csv(f"{RAW}/pffScoutingData.csv", usecols=["gameId", "playId", "nflId", "pff_role"])
    runners = pff[pff["pff_role"].str.lower() == "pass route"].merge(players, on="nflId")
    runners = runners[runners["officialPosition"].isin(["WR", "TE", "RB"])]
    runners = runners.merge(plays[["gameId", "playId"]], on=["gameId", "playId"])
    return plays, runners[["gameId", "playId", "nflId", "officialPosition", "displayName"]]


def process_game(path, plays_g, runners_g):
    t = pd.read_csv(path, usecols=COLS)
    t = t.merge(plays_g[["playId", "defensiveTeam"]], on="playId")  # pass plays only

    # Key frames per play (event is the string "None" on most rows)
    ev = t.drop_duplicates(["playId", "frameId"])[["playId", "frameId", "event"]]
    snap = ev[ev["event"] == "ball_snap"].groupby("playId")["frameId"].min().rename("snap")
    throw = ev[ev["event"].isin(THROW_EVENTS)].groupby("playId")["frameId"].min().rename("throw")
    kf = pd.concat([snap, throw], axis=1).dropna(subset=["snap"])  # skip plays with no snap
    kf["throw"] = kf["throw"].fillna(np.inf)
    kf["measure"] = np.minimum(kf["snap"] + MEASURE_FRAMES, kf["throw"])
    kf["wend"] = np.minimum(kf["snap"] + WINDOW_FRAMES, kf["throw"])
    t = t[t["team"] != "football"].merge(kf.reset_index(), on="playId")
    t["nflId"] = t["nflId"].astype(int)

    # Normalize direction so offense moves toward increasing x
    left = t["playDirection"] == "left"
    t.loc[left, "x"] = 120 - t.loc[left, "x"]
    t.loc[left, "y"] = 53.3 - t.loc[left, "y"]

    rr = t.merge(runners_g[KEY], on=KEY).sort_values(KEY + ["frameId"])
    de = t[t["team"] == t["defensiveTeam"]]
    if rr.empty:
        return None, None

    g = rr.groupby(KEY)["s"]
    rr["s_smooth"] = g.transform(lambda x: x.rolling(3, center=True, min_periods=1).mean())
    rr["ds"] = rr.groupby(KEY)["s_smooth"].diff() / 0.1
    rr["rel"] = rr["frameId"] - rr["snap"]

    routes = rr.groupby(KEY)[["snap", "wend"]].first()
    routes["route_top_speed"] = rr[rr["rel"] >= 0].groupby(KEY)["s"].max()
    routes["speed_1s"] = rr[rr["rel"] == ONE_SEC].set_index(KEY)["s_smooth"]
    routes["peak_accel_1s"] = rr[rr["rel"].between(1, ONE_SEC)].groupby(KEY)["ds"].max()

    win = rr[(rr["rel"] >= 0) & (rr["frameId"] <= rr["wend"])].copy()
    win["drop"] = win.groupby(KEY)["s_smooth"].cummax() - win["s_smooth"]
    routes["peak_decel"] = (-win[win["rel"] >= 1]["ds"]).groupby([win["playId"], win["nflId"]]).max()
    routes["speed_drop"] = win.groupby(KEY)["drop"].max()
    short = (routes["wend"] - routes["snap"]) < MIN_WINDOW
    routes.loc[short, ["peak_decel", "speed_drop"]] = np.nan

    # Separation: nearest defender at the measurement frame
    rm = rr[rr["frameId"] == rr["measure"]][KEY + ["frameId", "x", "y"]]
    dm = de[["playId", "frameId", "x", "y"]]
    pair = rm.merge(dm, on=["playId", "frameId"], suffixes=("", "_d"))
    pair["dist"] = np.hypot(pair["x"] - pair["x_d"], pair["y"] - pair["y_d"])
    routes["separation"] = pair.groupby(KEY)["dist"].min()

    # Speed curves: t = 0.0 to 3.0 s after the snap
    curves = rr[rr["rel"].between(0, WINDOW_FRAMES)][KEY + ["rel", "s_smooth"]].copy()
    curves["t"] = (curves["rel"] / 10).round(1)
    return routes.drop(columns=["snap", "wend"]).reset_index(), curves.drop(columns="rel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=None, help="only process the first N games (for testing)")
    args = ap.parse_args()

    start = time.time()
    plays, runners = load_lookups()
    files = sorted(glob.glob(f"{RAW}/tracking/tracking_*.csv"))[: args.n_games]

    all_routes, all_curves = [], []
    for i, path in enumerate(files, 1):
        gid = int(os.path.basename(path).split("_")[1].split(".")[0])
        r, c = process_game(path, plays[plays["gameId"] == gid], runners[runners["gameId"] == gid])
        if r is not None:
            r["gameId"], c["gameId"] = gid, gid
            all_routes.append(r)
            all_curves.append(c)
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)} games, {time.time() - start:.0f}s")

    routes = pd.concat(all_routes, ignore_index=True)
    routes = routes.merge(runners, on=["gameId", "playId", "nflId"]).merge(
        plays[["gameId", "playId", "week", "pff_passCoverageType", "passResult"]], on=["gameId", "playId"])
    routes = routes.rename(columns={"gameId": "game_id", "playId": "play_id", "nflId": "nfl_id",
                                    "displayName": "player_name", "officialPosition": "position",
                                    "pff_passCoverageType": "coverage_type", "passResult": "pass_result"})
    routes = routes[["game_id", "play_id", "nfl_id", "player_name", "position", "week", "coverage_type",
                     "pass_result", "separation", "route_top_speed", "speed_1s", "peak_accel_1s",
                     "peak_decel", "speed_drop"]]

    curves = pd.concat(all_curves, ignore_index=True).merge(
        plays[["gameId", "playId", "week"]], on=["gameId", "playId"])
    curves = curves.rename(columns={"gameId": "game_id", "playId": "play_id", "nflId": "nfl_id"})
    curves = curves[["game_id", "play_id", "nfl_id", "week", "t", "s_smooth"]]

    os.makedirs(OUT, exist_ok=True)
    routes.to_parquet(f"{OUT}/routes.parquet", index=False)
    curves.to_parquet(f"{OUT}/route_curves.parquet", index=False)
    print(f"routes.parquet: {len(routes):,} rows | route_curves.parquet: {len(curves):,} rows | "
          f"{len(files)} games | {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
