# Alex's notes for the README

## Method (one sentence)
From NFL tracking data we built a per-route speed curve for every wide receiver route on pass plays, summarized each WR's gas (speed and peak acceleration in the first second) and brakes (hardest deceleration and biggest speed drop in the first 3 s) over weeks 1–4, and predicted his average separation in weeks 5–8 with the same model (TabPFN) with and without those features, using 5-fold cross-validation across players repeated 5 times.

## Result (one sentence)
Gas and brakes help only slightly, and neither model predicts much: with TabPFN, adding them cut the error in predicting weeks 5–8 separation by 2.6% (mean MAE 0.335 yd with top speed alone vs. 0.326 yd with gas and brakes, full model better in 16 of 25 folds), but cross-validated R² stays below 0 for both and top speed is the only feature whose linear effect is clearly above zero (+0.07 yd per SD).

Chart note: `metric_vs_separation.png` uses `speed_drop`, the strongest brakes feature (Spearman 0.23 with weeks 5–8 separation) and nearly unrelated to top speed (Spearman −0.06), so it asks "is braking a different skill from speed?"

## Supporting numbers
- 106 WRs with 20+ routes in both weeks 1–4 and weeks 5–8 (34,478 WR/TE/RB routes built from all 122 games).
- For scale: guessing the average for every WR gives an MAE of 0.344 yd, so the full model is about 5% better than knowing nothing.
- Getting open is itself fairly stable: Spearman of weeks 1–4 separation with weeks 5–8 separation = 0.51, much higher than any speed feature (top speed 0.24, speed_drop 0.23, peak_decel 0.18).
- The result depends on the model: with the linear RidgeCV fallback the full model was 2.8% *worse* (wins 7 of 25 folds). TabPFN can use non-linear patterns that a linear model can't, which is where the small gain comes from. The linear BayesianRidge effects are the same under both.
- Robustness (TabPFN): with separation averaged over the 0.5 s up to the measurement frame (2.0–2.5 s after the snap, or ending at the throw) instead of a single frame, the full model is 3.1% better than baseline (MAE 0.323 vs. 0.334 yd, 16 of 25 folds) and 7.4% better than guessing the average, the same picture as the headline. The two separation measures agree almost perfectly across players (Spearman 0.97), so single-frame noise is not what limits accuracy.
- Man vs. zone (optional, run with RidgeCV): against man, nothing predicts separation (all effects ≈ 0); against zone, the full model ties the baseline (+0.9%, wins 72% of folds) with speed_drop the largest effect (+0.05 yd per SD, interval crosses 0). A hint, not a finding.

## Deviations from the framework
1. **TabPFN needs an API key.** TabPFN 9.2 requires a Prior Labs license acceptance and an API key set as the `TABPFN_TOKEN` environment variable. With it, both feature sets use `TabPFNRegressor()` (the framework's main model). Without it, `03_models.py` falls back to `RidgeCV(alphas=np.logspace(-3, 3, 13))` on standardized features for both sets, which gives the RidgeCV numbers above.
2. **Brakes features use the 90th percentile, not the median (Leo's review).** `peak_decel` and `speed_drop` are now the 90th percentile of a WR's weeks 1–4 routes, matching `top_speed`; `speed_1s` and `peak_accel_1s` stay medians. Reason: 69% of movement windows are cut at the throw (median 2.6 s), so many routes end before the break, and a median measured how often a WR breaks rather than how hard. This change was made after seeing the first (RidgeCV) result; we report both:
   - Median brakes (RidgeCV): full 4.0% worse than baseline, wins 24% of folds.
   - 90th-percentile brakes (RidgeCV): full 2.8% worse, wins 28% of folds.
   - 90th-percentile brakes (TabPFN, final): full 2.6% better, wins 64% of folds.
3. **Position group: WRs only, no position one-hot** (106 WRs ≥ 40, per the decision table).
4. Smoothing uses `min_periods=1` at the edges of each route so the first and last frames keep a value.

## How to run
1. Download the event data from github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data into `data/raw/` (so `data/raw/games.csv`, `data/raw/tracking/…`).
2. `pip install -r requirements.txt` (add `pip install tabpfn` and set `TABPFN_TOKEN` to reproduce the TabPFN numbers; without them the script uses RidgeCV).
3. From the repo root, run in order: `python src/01_build_routes.py` (about 15 s), `python src/02_player_metrics.py`, `python src/03_models.py` (about 2 min with TabPFN), `python src/04_charts.py`, `python src/05_speed_curves.py`.
