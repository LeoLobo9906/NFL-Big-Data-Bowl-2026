# Gas or Brakes: Does how a receiver stops predict how open he gets?

Leonardo Villalobos & Alex Oh · NFL Big Data Bowl NYU Edition (2021 season, weeks 1–8)

**What we built:** a **Brake Score** for NFL wide receivers, which measures how much speed a receiver drops when he cuts at the break, built from 2021 player tracking for 106 receivers in weeks 1–4, alongside his top speed and his burst off the line, and tested with a TabPFN machine-learning model whether these predict how open a receiver gets later in the season. 
**What it reveals:** adding braking and burst to top speed cut the error in predicting the NFL's official Next Gen Stats separation for weeks 9–18, games our model never saw, by 9.1% (better in 21 of 25 test splits), and by 2.6% on our own weeks 5–8 measure, while braking turned out to be a different trait from top speed (fast receivers are no more likely to brake hard). The best single predictor, though, is how open a receiver got early in the season, so getting open is a lasting skill that speed numbers only partly capture, and braking seems to help most against zone coverage, where a receiver can stop and sit in the hole between defenders. 
**Who would use it:** scouts can look at a receiver's Brake Score next to his top speed and his past separation when projecting who will get open, and coaches can see that against zone, routes with a hard break got the same receiver about 0.4 yd more space than his other routes. 
**Limitations:** the Brake Score is closely tied to how many breaking routes (curls, comebacks, outs) a receiver is asked to run (correlation 0.90), so it partly reflects his role in the offense, not just his skill, and the data has no route labels to separate the two; it is also one season and about 100 receivers.

**Terms used here**

- **Separation:** yards between the receiver and the nearest defender. More separation means more open.
- **Top speed:** how fast a receiver runs on his fastest routes (yards per second).
- **Burst (gas):** how fast he speeds up in the first second after the snap.
- **Brake Score (brakes):** how much speed he drops when he cuts, on his hardest-breaking routes.
- **Error (MAE):** how far off our prediction of a receiver's separation is, on average, in yards. Lower is better.
- **Test splits (folds):** we hide some receivers, train on the rest, and predict the hidden ones; repeating this 25 times shows whether a result is consistent.

## Models and accuracy at a glance

We predicted each receiver's later separation from his weeks 1–4 movement with the same model (TabPFN), once with top speed only and once with top speed plus gas and brakes, and tested both on 25 repeated splits where the model never sees the receivers it predicts.

| Test | Error, top speed only | Error, + gas & brakes | Improvement | Splits won |
|---|---|---|---|---|
| Our separation, weeks 5–8 (106 WRs) | 0.335 yd | 0.326 yd | 2.6% | 16 of 25 |
| Official NFL Next Gen Stats separation, weeks 9–18 (67 WRs) | 0.460 yd | 0.418 yd | 9.1% | 21 of 25 |

- **Fair comparison:** same model and same splits; only the features change.
- **No peeking ahead:** features come from weeks 1–4, and the targets come from later weeks.
- **Checked with a simpler linear model too:** the gain disappears on our own measure (2.8% worse) but holds on the official data (8.1% better); details in Methods.
- **Runs end to end:** a fresh clone runs `src/01`–`07` without errors.

## Same top speed, different brakes

![Speed curves](outputs/figures/speed_curves.png)

Each pair runs almost the same top speed but brakes very differently. The hard braker (orange) clearly slows down at the break about 2 seconds after the snap. Yet later in the season the light braker was just as open or more open in two of the three pairs, so braking alone doesn't decide who gets open. Cooper Kupp, in his 2021 triple-crown season, barely brakes at all: he wins at the break without stopping.

## Does adding gas and brakes beat top speed?

![Model comparison](outputs/figures/model_comparison.png)

On our own weeks 5–8 measure the gain is small (2.6%); on the NFL's official separation for weeks 9–18 it is larger (9.1%, 21 of 25 splits). See "Models and accuracy at a glance" above.

## Brake Score (speed drop) in weeks 1–4 vs. separation in weeks 5–8

![Metric vs separation](outputs/figures/metric_vs_separation.png)

## Brake Score top 10

| # | Receiver | Brake Score, wks 1–4 (yd/s) | Top speed, wks 1–4 (yd/s) | Separation, wks 5–8 (yd) | Separation vs. zone, wks 5–8 (yd) |
|---|---|---|---|---|---|
| 1 | Jarvis Landry | 4.48 | 7.73 | 2.71 | 2.95 |
| 2 | Jamison Crowder | 4.26 | 8.57 | 3.20 | 3.82 |
| 3 | Sterling Shepard | 3.85 | 7.84 | 3.21 | 3.40 |
| 4 | Darius Slayton | 3.84 | 8.43 | 2.61 | 3.32 |
| 5 | Laviska Shenault | 3.83 | 8.11 | 3.07 | 3.62 |
| 6 | Amon-Ra St. Brown | 3.80 | 8.27 | 3.00 | 3.23 |
| 7 | Greg Ward | 3.68 | 7.70 | 2.13 | 2.29 |
| 8 | Tyreek Hill | 3.67 | 9.10 | 3.19 | 3.65 |
| 9 | Chester Rogers | 3.64 | 8.03 | 2.71 | 3.51 |
| 10 | D.J. Moore | 3.52 | 8.71 | 3.11 | 3.41 |
| | **Median of all 106 WRs** | **2.58** | **8.43** | **2.87** | **3.31** |

The hardest brakers are mostly not burners: 7 of the 10 are at or below the median top speed (Hill, Moore and Crowder are the exceptions). 7 of the 10 were more open than the median receiver against zone later in the season, and 6 of 10 overall. Many are slot receivers (Landry, Crowder, Shepard, St. Brown, Ward, Rogers), who run more breaking routes, which fits the route-mix limitation above. Brake Score = for each route, the biggest drop in speed from his fastest point so far, within 3 s of the snap (or up to the throw); then his 90th-percentile route over weeks 1–4. All numbers are from `outputs/tables/player_metrics.csv`.

## Methods and assumptions

- **Players:** WRs only (106), with at least 20 routes in weeks 1–4 and 20 in weeks 5–8; pass plays only (complete, incomplete, intercepted).
- **Features (weeks 1–4):** `top_speed` (90th percentile of per-route max speed); gas: `speed_1s`, `peak_accel_1s` (medians); brakes: `peak_decel`, `speed_drop` (90th percentiles, because 69% of routes are cut short by the throw, a median measured how often a WR breaks, not how hard; this choice was made after a first look at results; see below).
- **Target (weeks 5–8):** `avg_separation`, mean yards to the nearest defender at 2.5 s after the snap or at the throw.
- **Exploratory (run after the main results, so treat as a lead, not a finding):** with the linear model, top speed + Brake Score alone beats top speed in 16 of 25 folds (1.7% better), while top speed + burst features is 4.0% worse (1 of 25), so on our own separation measure braking carries the signal and burst adds noise. On official NGS separation (below) it's the other way around: burst matters at least as much as braking. **Our reading (not tested):** the two checks measure different moments. Ours is 2.5 s after the snap, around the break, where stopping creates space; the official number is taken when the ball arrives, where burst and getting downfield matter more. A "separation gained at the break" metric went the wrong way (Spearman −0.23), likely because only tightly covered receivers gain much at the cut. Splitting WRs at the median top speed and Brake Score, receivers who combine speed with hard brakes, like Calvin Ridley and D.J. Moore, got the most open in weeks 5–8 (3.16 yd vs. 2.75 for slower, soft brakers), and at either speed level hard brakers were more open, most of all against zone (3.61 vs. 3.19 yd; hard vs. soft brakers against zone +0.29 yd, Mann-Whitney p = 0.001, uncorrected). On individual routes (`src/07_break_routes.py`; 12,001 WR routes thrown 2.5 s or more after the snap, so all measured at the same moment), routes with a hard break had more separation, most clearly against zone (+0.46 yd, 95% CI +0.35 to +0.57) and less against man (+0.30 yd; the man result depends on which receivers are included). This fits the coaching rule that break routes beat zone, and it explains the route-mix finding: much of the Brake Score's link to separation comes from the type of route. **Next steps:** compare braking within the same route type (which needs route labels), and separate the effects of braking and burst using more seasons.
- **Out-of-sample check on official NFL data** (`src/06_ngs_check.py`): the same weeks 1–4 features, folds and TabPFN model predict official Next Gen Stats separation (yards to the nearest defender when the ball arrives, targeted throws only) in weeks 9–18 of 2021, from nflverse. For 67 WRs with 20+ targets, gas and brakes beat top speed alone by 9.1% (MAE 0.418 vs. 0.460 yd, 21 of 25 folds; 8.1% with RidgeCV) and beat guessing the average by 6.4%, while top speed alone did worse than guessing. Spearman with later NGS separation: burst 0.34–0.37, brakes 0.28, top speed 0.21. Limits: targeted throws only, so it leans toward the QB's favorite options; 67 players; no man/zone split, so it doesn't test the zone hint.
- **Route mix:** the Brake Score correlates 0.90 with `share_break_routes` (the share of a receiver's routes with a hard break), partly because both are built from the same speed drops. Holding route mix constant, its link to later separation disappears (−0.11), so we can't tell how much of the Brake Score is skill and how much is the routes he's asked to run.
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
2. `pip install -r requirements.txt` (includes `tabpfn`), plus a Prior Labs API key in the `TABPFN_TOKEN` environment variable (without it, `03_models.py` falls back to RidgeCV and gives the RidgeCV rows above).
3. From the repo root, run in order:
   ```
   python src/01_build_routes.py
   python src/02_player_metrics.py
   python src/03_models.py
   python src/04_charts.py
   python src/05_speed_curves.py
   python src/06_ngs_check.py   # optional: downloads public NGS data from nflverse
   python src/07_break_routes.py  # optional: route-level check, prints only
   ```
   If `06` fails with an SSL certificate error (common with python.org Python on macOS), download the file first: `curl -L -o data/raw/ngs_2021_receiving.csv.gz https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_2021_receiving.csv.gz`
