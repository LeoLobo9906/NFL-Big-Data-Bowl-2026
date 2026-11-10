# Gas or Brakes: Does how a receiver stops predict how open he gets?

Leonardo Villalobos & Alex Oh · NFL Big Data Bowl NYU Edition (2021 season, weeks 1–8)

**Top speed is enough, and neither predicts much: adding a wide receiver's burst and braking to his top speed made our forecast of his weeks 5–8 separation 2.8% worse (MAE 0.341 vs. 0.332 yards, better in only 7 of 25 folds), while his own weeks 1–4 separation tracked it twice as well as top speed did (Spearman 0.51 vs. 0.24).** From player tracking we built a speed curve for every route run by 106 WRs with 20+ routes in each half, measured "gas" (speed and peak acceleration in the first second) and "brakes" (hardest deceleration and biggest speed drop in the first 3 seconds, 90th percentile of his routes), and compared the same ridge regression with and without them using 5-fold cross-validation across players, repeated 5 times. The one hint is against zone, where a receiver's speed drop ranks his later separation better than top speed (Spearman 0.31 vs. 0.16), which fits sitting down in a hole between defenders, but its model effect isn't clear of zero. For scouts and coaches, the takeaway is to project who gets open from a receiver's own separation history rather than from his speed or how hard he stops, because getting open is a stable skill these movement numbers miss. Separation is measured to the nearest defender 2.5 s after the snap (or at the throw if earlier), braking depends on the routes a receiver is asked to run, and the sample is eight weeks of one season.

## Same top speed, different brakes

![Speed curves](outputs/figures/speed_curves.png)

Each pair has nearly the same top speed but very different braking. The hard braker (orange) clearly slows at the break around 2 s, yet in weeks 5–8 the light braker was as open or more open in two of three pairs. Cooper Kupp, in his 2021 triple-crown season, barely brakes at all: he wins at the break without stopping.

## Does adding gas and brakes beat top speed?

![Model comparison](outputs/figures/model_comparison.png)

## Brakes in weeks 1–4 vs. separation in weeks 5–8

![Metric vs separation](outputs/figures/metric_vs_separation.png)

## Methods and assumptions

- **Players:** WRs only (106), with at least 20 routes in weeks 1–4 and 20 in weeks 5–8; pass plays only (complete, incomplete, intercepted).
- **Features (weeks 1–4):** `top_speed` (90th percentile of per-route max speed); gas: `speed_1s`, `peak_accel_1s` (medians); brakes: `peak_decel`, `speed_drop` (90th percentiles, because 69% of routes are cut short by the throw, a median measured how often a WR breaks, not how hard; with medians the full model was 4.0% worse, so the conclusion is the same).
- **Target (weeks 5–8):** `avg_separation`, mean yards to the nearest defender at 2.5 s after the snap or at the throw.
- **Model:** `RidgeCV` on standardized features for both feature sets (the framework's fallback; TabPFN needs a license key). Cross-validated R² is below zero for both, so speed features alone explain little of who gets open.
- Full results: `outputs/tables/` (`model_results.csv`, `feature_weights.csv`, `predictions.csv`, `player_metrics.csv`).

## How to run

1. Download the event data from [nfl-big-data-bowl-regional-event-data](https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data) into `data/raw/` (so `data/raw/games.csv`, `data/raw/tracking/…`).
2. `pip install -r requirements.txt`
3. From the repo root, run in order:
   ```
   python src/01_build_routes.py
   python src/02_player_metrics.py
   python src/03_models.py
   python src/04_charts.py
   python src/05_speed_curves.py
   ```
