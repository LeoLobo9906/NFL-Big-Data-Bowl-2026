# Gas or Brakes: Does how a receiver stops predict how open he gets?

Leonardo Villalobos & Alex Oh · NFL Big Data Bowl NYU Edition (2021 season, weeks 1–8)

**What we built:** a **Brake Score** for NFL wide receivers, how much speed a receiver sheds at the break (90th percentile of his biggest speed drop in the first 3 s after the snap), plus a burst measure, from 2021 player tracking for 106 WRs over weeks 1–4. **What it reveals:** braking is a separate skill from speed (Brake Score vs. top speed: Spearman −0.06), and adding braking and burst to top speed cut the error in predicting a receiver's weeks 5–8 separation by 2.6% (0.326 vs. 0.335 yards, better in 16 of 25 cross-validation folds with TabPFN) and the error in predicting the NFL's official Next Gen Stats separation in weeks 9–18, games outside our data, by 9.1% (67 WRs with 20+ targets, 21 of 25 folds), though his own early-season separation remained the best guide (Spearman 0.51 vs. 0.24 for top speed). The Brake Score may matter most against zone, where it ranks later separation better than top speed does (Spearman 0.31 vs. 0.16), which fits a receiver sitting down in a hole between defenders, but that effect is not yet clear of zero. **Who would use it:** scouts and coaches can rank receivers by Brake Score alongside top speed and separation history when projecting who will get open, because it captures something speed misses. **Limitations:** separation is measured to the nearest defender 2.5 s after the snap (or at the throw if earlier), braking depends on the routes a receiver is asked to run, and the sample is eight weeks of one season.

## Same top speed, different brakes

![Speed curves](outputs/figures/speed_curves.png)

Each pair has nearly the same top speed but very different braking. The hard braker (orange) clearly slows at the break around 2 s, yet in weeks 5–8 the light braker was as open or more open in two of three pairs: braking alone doesn't decide who gets open. Cooper Kupp, in his 2021 triple-crown season, barely brakes at all: he wins at the break without stopping.

## Does adding gas and brakes beat top speed?

![Model comparison](outputs/figures/model_comparison.png)

## Brake Score (speed drop) in weeks 1–4 vs. separation in weeks 5–8

![Metric vs separation](outputs/figures/metric_vs_separation.png)

## Methods and assumptions

- **Players:** WRs only (106), with at least 20 routes in weeks 1–4 and 20 in weeks 5–8; pass plays only (complete, incomplete, intercepted).
- **Features (weeks 1–4):** `top_speed` (90th percentile of per-route max speed); gas: `speed_1s`, `peak_accel_1s` (medians); brakes: `peak_decel`, `speed_drop` (90th percentiles, because 69% of routes are cut short by the throw, a median measured how often a WR breaks, not how hard; this choice was made after a first look at results; see below).
- **Target (weeks 5–8):** `avg_separation`, mean yards to the nearest defender at 2.5 s after the snap or at the throw.
- **Exploratory (run after the main results, so treat as a lead, not a finding):** with the linear model, top speed + Brake Score alone beats top speed in 16 of 25 folds (1.7% better), while top speed + burst features is 4.0% worse (1 of 25), so on our own separation measure braking carries the signal and burst adds noise (on official NGS separation, below, burst correlates at least as strongly as braking, so the two are hard to separate). A "separation gained at the break" metric went the wrong way (Spearman −0.23), likely because only tightly covered receivers gain much at the cut. Splitting WRs at the median top speed and Brake Score, receivers who combine speed with hard brakes, like Calvin Ridley and D.J. Moore, got the most open in weeks 5–8 (3.16 yd vs. 2.75 for slower, soft brakers), and at either speed level hard brakers were more open, most of all against zone (3.61 vs. 3.19 yd; hard vs. soft brakers against zone +0.29 yd, Mann-Whitney p = 0.001, uncorrected). **Next steps:** model braking without the burst features and test it on more seasons.
- **Out-of-sample check on official NFL data** (`src/06_ngs_check.py`): the same weeks 1–4 features, folds and TabPFN model predict official Next Gen Stats separation (yards to the nearest defender when the ball arrives, targeted throws only) in weeks 9–18 of 2021, from nflverse. For 67 WRs with 20+ targets, gas and brakes beat top speed alone by 9.1% (MAE 0.418 vs. 0.460 yd, 21 of 25 folds; 8.1% with RidgeCV) and beat guessing the average by 6.4%, while top speed alone did worse than guessing. Spearman with later NGS separation: burst 0.34–0.37, brakes 0.28, top speed 0.21. Limits: targeted throws only, so it leans toward the QB's favorite options; 67 players; no man/zone split, so it doesn't test the zone hint.
- **Robustness:** the result holds when separation is averaged over 2.0–2.5 s instead of one frame (3.1% better, 16 of 25 folds); the two measures agree almost perfectly across players (Spearman 0.97).
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
   python src/06_ngs_check.py   # optional: downloads public NGS data from nflverse
   ```
