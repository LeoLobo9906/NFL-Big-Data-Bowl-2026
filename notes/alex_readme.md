# Alex's notes for the README

## Method (one sentence)
From NFL tracking data we built a per-route speed curve for every wide receiver route on pass plays, summarized each WR's gas (speed and peak acceleration in the first second) and brakes (hardest deceleration and biggest speed drop in the first 3 s) over weeks 1–4, and predicted his average separation in weeks 5–8 with the same model with and without those features, using 5-fold cross-validation across players repeated 5 times.

## Result (one sentence)
Top speed is enough, and neither predicts much: adding gas and brakes did not improve the prediction of future separation (mean MAE 0.332 yd with top speed alone vs. 0.341 yd with gas and brakes, 2.8% worse, full model better in only 28% of 25 folds; cross-validated R² below 0 for both), and top speed was the only feature whose effect was clearly above zero (+0.07 yd per SD).

Chart note: `metric_vs_separation.png` uses `speed_drop`, the strongest brakes feature (Spearman 0.23 with weeks 5–8 separation) and nearly unrelated to top speed (Spearman −0.06), so it asks "is braking a different skill from speed?"

## Supporting numbers
- 106 WRs with 20+ routes in both weeks 1–4 and weeks 5–8 (34,478 WR/TE/RB routes built from all 122 games).
- Getting open is itself fairly stable: Spearman of weeks 1–4 separation with weeks 5–8 separation = 0.51, much higher than any speed feature (top speed 0.24, speed_drop 0.23, peak_decel 0.18).
- Neither model explains much: cross-validated R² is below 0 for both, so movement style and top speed alone explain little of who gets open.
- Man vs. zone (optional): against man, nothing predicts separation (all effects ≈ 0); against zone, the full model ties the baseline (+0.9%, wins 72% of folds) with speed_drop the largest effect (+0.05 yd per SD, interval crosses 0). A hint, not a finding.

## Deviations from the framework
1. **RidgeCV instead of TabPFN.** TabPFN 9.2 requires a Prior Labs license acceptance and API token before it can download weights, so it failed to load. Per the framework, both feature sets use `RidgeCV(alphas=np.logspace(-3, 3, 13))` on standardized features.
2. **Brakes features use the 90th percentile, not the median (Leo's review).** `peak_decel` and `speed_drop` are now the 90th percentile of a WR's weeks 1–4 routes, matching `top_speed`; `speed_1s` and `peak_accel_1s` stay medians. Reason: 69% of movement windows are cut at the throw (median 2.6 s), so many routes end before the break, and a median measured how often a WR breaks rather than how hard. This change was made after seeing the first result; we report both:
   - Median brakes: full 4.0% worse than baseline, wins 24% of folds.
   - 90th-percentile brakes (final): full 2.8% worse, wins 28% of folds.
   Same conclusion either way.
3. **Position group: WRs only, no position one-hot** (106 WRs ≥ 40, per the decision table).
4. Smoothing uses `min_periods=1` at the edges of each route so the first and last frames keep a value.

## How to run
1. Download the event data from github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data into `data/raw/` (so `data/raw/games.csv`, `data/raw/tracking/…`).
2. `pip install -r requirements.txt`
3. From the repo root, run in order: `python src/01_build_routes.py` (about 15 s), `python src/02_player_metrics.py`, `python src/03_models.py`, `python src/04_charts.py`, `python src/05_speed_curves.py`.
