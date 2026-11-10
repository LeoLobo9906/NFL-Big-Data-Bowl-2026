# Gas or Brakes: Does how a receiver stops predict how open he gets?

Leonardo Villalobos & Alex Oh · NFL Big Data Bowl NYU Edition (2021 season, weeks 1–8)

**What we built:** a "gas and brakes" profile for 106 NFL wide receivers from 2021 player tracking, measuring how fast each one bursts off the line in the first second and how hard he brakes at the break, over every route in weeks 1–4. **What it reveals:** getting open is a stable skill that speed barely explains, since a receiver's early-season separation tracked his weeks 5–8 separation twice as well as his top speed did (Spearman 0.51 vs. 0.24), and adding gas and brakes to top speed cut prediction error by only 2.6% (0.326 vs. 0.335 yards, better in 16 of 25 cross-validation folds with TabPFN). Braking shows up most against zone, where a receiver's speed drop at the break ranks his later separation better than top speed (Spearman 0.31 vs. 0.16), which fits sitting down in a hole between defenders, though it is a hint rather than a proven effect. **Who would use it:** scouts and coaches projecting which receivers will get open should start from separation history and use braking alongside top speed, not top speed alone. **Limitations:** separation is measured to the nearest defender 2.5 s after the snap (or at the throw if earlier), braking depends on the routes a receiver is asked to run, and the sample is eight weeks of one season.

## Same top speed, different brakes

![Speed curves](outputs/figures/speed_curves.png)

Each pair has nearly the same top speed but very different braking. The hard braker (orange) clearly slows at the break around 2 s, yet in weeks 5–8 the light braker was as open or more open in two of three pairs: braking alone doesn't decide who gets open. Cooper Kupp, in his 2021 triple-crown season, barely brakes at all: he wins at the break without stopping.

## Does adding gas and brakes beat top speed?

![Model comparison](outputs/figures/model_comparison.png)

## Brakes in weeks 1–4 vs. separation in weeks 5–8

![Metric vs separation](outputs/figures/metric_vs_separation.png)

## Methods and assumptions

- **Players:** WRs only (106), with at least 20 routes in weeks 1–4 and 20 in weeks 5–8; pass plays only (complete, incomplete, intercepted).
- **Features (weeks 1–4):** `top_speed` (90th percentile of per-route max speed); gas: `speed_1s`, `peak_accel_1s` (medians); brakes: `peak_decel`, `speed_drop` (90th percentiles, because 69% of routes are cut short by the throw, a median measured how often a WR breaks, not how hard; this choice was made after a first look at results; see below).
- **Target (weeks 5–8):** `avg_separation`, mean yards to the nearest defender at 2.5 s after the snap or at the throw.
- **Model:** `TabPFNRegressor()` for both feature sets. Cross-validated R² is below zero for both (guessing the average gives MAE 0.344 yd), so speed features alone explain little of who gets open.
- **The small gain depends on the model, so we report every version we ran:**

  | Brakes summary | Model | Full vs. baseline MAE | Folds full wins |
  |---|---|---|---|
  | Median | RidgeCV | 4.0% worse | 6 of 25 |
  | 90th percentile | RidgeCV | 2.8% worse | 7 of 25 |
  | 90th percentile | **TabPFN (final)** | **2.6% better** | **16 of 25** |

  TabPFN can use non-linear patterns a linear model can't; in a linear model only top speed has an effect clearly above zero (+0.07 yd per SD).
- Full results: `outputs/tables/` (`model_results.csv`, `feature_weights.csv`, `predictions.csv`, `player_metrics.csv`).

## How to run

1. Download the event data from [nfl-big-data-bowl-regional-event-data](https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data) into `data/raw/` (so `data/raw/games.csv`, `data/raw/tracking/…`).
2. `pip install -r requirements.txt`, plus `pip install tabpfn` and a Prior Labs API key in the `TABPFN_TOKEN` environment variable (without it, `03_models.py` falls back to RidgeCV and gives the RidgeCV rows above).
3. From the repo root, run in order:
   ```
   python src/01_build_routes.py
   python src/02_player_metrics.py
   python src/03_models.py
   python src/04_charts.py
   python src/05_speed_curves.py
   ```
