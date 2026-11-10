"""03_models.py: baseline (top_speed) vs. full (top_speed + gas + brakes) model.

Reads outputs/tables/player_metrics.csv, writes:
  outputs/tables/model_results.csv   (one row per fold per feature set)
  outputs/tables/predictions.csv     (out-of-fold, first repeat)
  outputs/tables/feature_weights.csv (BayesianRidge effects + Spearman)
Main model: TabPFNRegressor; falls back to RidgeCV on standardized features
(for both sets) if TabPFN fails to load or a fit takes over a minute.
Usage (from repo root):  python src/03_models.py [--target avg_sep_man_w5_8]
  A non-default --target only prints the summary; it writes no files.
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
OUT = "outputs/tables"
BASELINE = ["top_speed"]
FULL = ["top_speed", "speed_1s", "peak_accel_1s", "peak_decel", "speed_drop"]
SETS = {"baseline": BASELINE, "full": FULL}


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


def pick_model(X, y):
    """Return (factory, name). Try TabPFN once; fall back to RidgeCV."""
    try:
        from tabpfn import TabPFNRegressor
        start = time.time()
        TabPFNRegressor().fit(X, y).predict(X[:5])
        took = time.time() - start
        if took > 60:
            print(f"TabPFN took {took:.0f}s for one fit (>60s) -> using RidgeCV for both sets")
            return ridge, "RidgeCV"
        return TabPFNRegressor, "TabPFN"
    except Exception as e:
        print(f"TabPFN failed to load/fit ({type(e).__name__}: {str(e).splitlines()[0][:80]}) -> using RidgeCV for both sets")
        return ridge, "RidgeCV"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="avg_separation")
    args = ap.parse_args()
    target = args.target
    write = target == "avg_separation"

    m = pd.read_csv(f"{OUT}/player_metrics.csv").dropna(subset=FULL + [target]).reset_index(drop=True)
    y = m[target].to_numpy()
    factory, model_name = pick_model(m[FULL].to_numpy(), y)

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    rows = []
    preds = {name: np.full(len(m), np.nan) for name in SETS}
    for i, (tr, te) in enumerate(rkf.split(m)):
        repeat, fold = i // 5 + 1, i % 5 + 1
        for name, cols in SETS.items():
            X = m[cols].to_numpy()
            p = factory().fit(X[tr], y[tr]).predict(X[te])
            rows.append({"repeat": repeat, "fold": fold, "feature_set": name, "n_train": len(tr),
                         "n_test": len(te), "mae_yards": mean_absolute_error(y[te], p), "r2": r2_score(y[te], p)})
            if repeat == 1:
                preds[name][te] = p
    res = pd.DataFrame(rows)

    # Summary
    mae = res.groupby("feature_set")["mae_yards"].mean()
    r2 = res.groupby("feature_set")["r2"].mean()
    wide = res.pivot_table(index=["repeat", "fold"], columns="feature_set", values="mae_yards")
    win_share = (wide["full"] < wide["baseline"]).mean()
    improve = (mae["baseline"] - mae["full"]) / mae["baseline"] * 100

    # Explaining model: BayesianRidge on z-scored full features, all players
    Z = StandardScaler().fit_transform(m[FULL])
    br = BayesianRidge().fit(Z, y)
    se = np.sqrt(np.diag(br.sigma_))
    fw = pd.DataFrame({
        "feature": FULL,
        "effect_per_sd": br.coef_,
        "ci_low": br.coef_ - 1.645 * se,
        "ci_high": br.coef_ + 1.645 * se,
        "spearman_top_speed": [m[f].corr(m["top_speed"], method="spearman") for f in FULL],
        "spearman_separation": [m[f].corr(m[target], method="spearman") for f in FULL],
    })
    top = fw.loc[fw["effect_per_sd"].abs().idxmax()]

    print(f"\n=== Target: {target} | model: {model_name} | {len(m)} players | 5x5 repeated CV ===")
    print(f"Mean MAE  baseline: {mae['baseline']:.3f} yd   full: {mae['full']:.3f} yd")
    print(f"Mean R2   baseline: {r2['baseline']:.3f}      full: {r2['full']:.3f}")
    print(f"Improvement: {improve:+.1f}%   full wins {win_share:.0%} of 25 folds")
    print("\nBayesianRidge effect per SD (90% interval):")
    for _, r in fw.iterrows():
        clear = "clear of 0" if r.ci_low > 0 or r.ci_high < 0 else "crosses 0"
        print(f"  {r.feature:<14}{r.effect_per_sd:+.3f}  [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  {clear}")
    print(f"Top effect: {top.feature} ({top.effect_per_sd:+.3f} yd per SD)")
    if target == "avg_separation":
        print(f"Context: Spearman sep_w1_4 vs avg_separation = "
              f"{m['sep_w1_4'].corr(m['avg_separation'], method='spearman'):.2f}")

    print("\nFlags:")
    flags = []
    if win_share < 0.5:
        flags.append(f"full wins only {win_share:.0%} of folds -> report 'top speed is enough'")
    if max(r2) > 0.5:
        flags.append(f"suspiciously high R2 ({max(r2):.2f}) -> check for leakage")
    for _, r in fw.iterrows():
        if r.feature in ("speed_1s", "peak_accel_1s", "peak_decel", "speed_drop", "top_speed") and r.effect_per_sd < 0 \
                and r.ci_high < 0:
            flags.append(f"{r.feature} has a clearly negative effect -> check football sense")
    print("  " + ("\n  ".join(flags) if flags else "none"))

    if write:
        res.to_csv(f"{OUT}/model_results.csv", index=False)
        fw.to_csv(f"{OUT}/feature_weights.csv", index=False)
        pd.DataFrame({
            "nfl_id": m["nfl_id"], "player_name": m["player_name"], "position": m["position"],
            "actual": y, "pred_baseline": preds["baseline"], "pred_full": preds["full"],
            "residual_full": y - preds["full"],
        }).to_csv(f"{OUT}/predictions.csv", index=False)
        print("\nWrote model_results.csv, feature_weights.csv, predictions.csv")


if __name__ == "__main__":
    main()
