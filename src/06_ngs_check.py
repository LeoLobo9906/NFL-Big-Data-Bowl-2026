"""06_ngs_check.py: supplemental check against official NFL Next Gen Stats separation.

Do our weeks 1-4 metrics (outputs/tables/player_metrics.csv) predict official NGS
separation (yards to nearest defender when the ball arrives, targeted throws only)
in weeks 9-18 of 2021, games outside the event data? Public data from nflverse;
use approved by the event judges. Same feature sets, splits and model as 03.

Reads outputs/tables/player_metrics.csv and data/raw/ngs_2021_receiving.csv.gz
(downloaded automatically if missing). Writes outputs/tables/ngs_check.csv.
Usage (from repo root):  python src/06_ngs_check.py
"""
import importlib.util
import os
import re
import urllib.request
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold

warnings.filterwarnings("ignore")
NGS_URL = "https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_2021_receiving.csv.gz"
NGS_PATH = "data/raw/ngs_2021_receiving.csv.gz"
MIN_TARGETS = 20

# Reuse the feature sets and model choice (TabPFN, RidgeCV fallback) from 03_models.py
spec = importlib.util.spec_from_file_location("models", "src/03_models.py")
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


def norm(name):
    return re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name.lower()))


def main():
    if not os.path.exists(NGS_PATH):
        print(f"Downloading {NGS_URL}")
        try:
            urllib.request.urlretrieve(NGS_URL, NGS_PATH)
        except Exception as e:  # e.g. SSL certificate error on python.org Python for macOS
            print(f"Download failed ({type(e).__name__}). Run this from the repo root, then rerun:\n"
                  f"  curl -L -o {NGS_PATH} {NGS_URL}")
            raise SystemExit(1)
    ngs = pd.read_csv(NGS_PATH)
    ngs = ngs[(ngs["season_type"] == "REG") & ngs["week"].between(9, 18)].copy()
    ngs["key"] = ngs["player_display_name"].map(norm)
    ngs["sep_x_targets"] = ngs["avg_separation"] * ngs["targets"]
    g = ngs.groupby("key").agg(sxt=("sep_x_targets", "sum"), targets=("targets", "sum"))
    g["ngs_sep_w9_18"] = g["sxt"] / g["targets"]   # target-weighted average

    m = pd.read_csv("outputs/tables/player_metrics.csv")
    m["key"] = m["player_name"].map(norm)
    m = m.join(g[["ngs_sep_w9_18", "targets"]], on="key")
    matched = m["ngs_sep_w9_18"].notna().sum()
    v = m[m["targets"] >= MIN_TARGETS].reset_index(drop=True)
    y = v["ngs_sep_w9_18"].to_numpy()
    factory, model_name = models.pick_model(v[models.FULL].to_numpy(), y)

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    mae = {"guess_average": [], "baseline": [], "full": []}
    r2 = {"baseline": [], "full": []}
    for tr, te in rkf.split(v):
        mae["guess_average"].append(mean_absolute_error(y[te], np.full(len(te), y[tr].mean())))
        for name, cols in models.SETS.items():
            X = v[cols].to_numpy()
            p = factory().fit(X[tr], y[tr]).predict(X[te])
            mae[name].append(mean_absolute_error(y[te], p))
            r2[name].append(r2_score(y[te], p))

    base = np.array(mae["baseline"])
    rows = []
    for name, e in mae.items():
        e = np.array(e)
        rows.append({"feature_set": name, "model": "mean" if name == "guess_average" else model_name,
                     "n_players": len(v), "mae_yards": e.mean(),
                     "r2": np.mean(r2[name]) if name in r2 else np.nan,
                     "pct_vs_baseline": (base.mean() - e.mean()) / base.mean() * 100,
                     "folds_beating_baseline": int((e < base).sum()) if name != "baseline" else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv("outputs/tables/ngs_check.csv", index=False)

    print(f"\n=== NGS separation weeks 9-18 | {model_name} | {matched} of {len(m)} WRs matched, "
          f"{len(v)} with {MIN_TARGETS}+ targets | 5x5 CV ===")
    print(out.round(3).to_string(index=False))
    print("\nSpearman with NGS separation, weeks 9-18:")
    for f in models.FULL + ["sep_w1_4"]:
        print(f"  {f:<15}{v[f].corr(v['ngs_sep_w9_18'], method='spearman'):+.2f}")
    print("\nWrote outputs/tables/ngs_check.csv")


if __name__ == "__main__":
    main()
