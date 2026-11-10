"""07_break_routes.py: exploratory, read-only route-level check (writes no files).

Do WR routes with a hard break (speed_drop > 2 yd/s, same threshold as
share_break_routes) get more separation than routes without one, against man
and against zone? Uses only plays thrown 2.5 s or later after the snap, so every
route's separation is measured at the same moment (snap + 2.5 s).
Prints pooled gaps (95% CI) and within-player gaps (each WR vs. himself), with
no route minimum and with 5+ routes of each kind per WR.

Reads data/processed/routes.parquet and data/raw/tracking/*.csv (for throw time).
Usage (from repo root):  python src/07_break_routes.py
"""
import glob

import numpy as np
import pandas as pd
from scipy import stats

THROW_EVENTS = ["pass_forward", "autoevent_passforward"]


def throw_times():
    out = []
    for f in sorted(glob.glob("data/raw/tracking/tracking_*.csv")):
        t = pd.read_csv(f, usecols=["gameId", "playId", "frameId", "event"])
        ev = t.drop_duplicates(["playId", "frameId"])
        snap = ev[ev["event"] == "ball_snap"].groupby("playId")["frameId"].min()
        throw = ev[ev["event"].isin(THROW_EVENTS)].groupby("playId")["frameId"].min()
        d = (throw - snap).dropna().rename("throw_frames").reset_index()
        d["game_id"] = t["gameId"].iloc[0]
        out.append(d.rename(columns={"playId": "play_id"}))
    return pd.concat(out)


def main():
    r = pd.read_parquet("data/processed/routes.parquet").merge(throw_times(), on=["game_id", "play_id"])
    r = r[(r["position"] == "WR") & r["speed_drop"].notna() & r["coverage_type"].isin(["Man", "Zone"])
          & (r["throw_frames"] >= 25)].copy()
    r["kind"] = np.where(r["speed_drop"] > 2, "break", "no break")
    print(f"{len(r):,} WR routes on plays thrown at 2.5 s or later ({r['nfl_id'].nunique()} WRs)\n")
    print(r.groupby(["coverage_type", "kind"])["separation"].agg(["size", "mean"]).round(2).to_string(), "\n")

    for cov in ["Zone", "Man"]:
        d = r[r["coverage_type"] == cov]
        a = d.loc[d["kind"] == "break", "separation"]
        b = d.loc[d["kind"] == "no break", "separation"]
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        print(f"{cov}: pooled break minus no-break = {diff:+.2f} yd (95% CI {diff - 1.96 * se:+.2f} to {diff + 1.96 * se:+.2f})")
        p = d.groupby(["nfl_id", "kind"])["separation"].agg(["mean", "size"]).unstack()
        for k in [1, 5]:
            q = p[(p["size"] >= k).all(axis=1)]
            w = q["mean"]["break"] - q["mean"]["no break"]
            print(f"  within player, {k}+ routes of each kind: {w.mean():+.2f} yd across {len(w)} WRs, "
                  f"{(w > 0).mean():.0%} positive, Wilcoxon p = {stats.wilcoxon(w).pvalue:.3g}")
    print("\nExploratory: found after the main results; 'break' partly reflects route type (route design, not proof of cause).")


if __name__ == "__main__":
    main()
