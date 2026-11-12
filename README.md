# Gas or Brakes 🏈
### Does how an NFL receiver *stops* predict how open he gets?

**Leonardo Villalobos & Alex Oh** · NFL Big Data Bowl, NYU Edition · Python · pandas · scikit-learn · TabPFN · matplotlib

![Speed curves](outputs/figures/speed_curves.png)

## TL;DR

- We built the **Brake Score**, a new metric from raw NFL player-tracking data (10 frames per second) that measures how sharply a wide receiver slows down when he cuts.
- Adding the Brake Score and a burst metric to top speed **reduced error by 9.1%** when predicting the NFL's official **Next Gen Stats** separation for weeks 9–18. Those games were completely outside our dataset, and the full model won **21 of 25** cross-validation splits.
- Braking is a **separate trait from speed**: fast receivers are no more likely to brake hard. It helps most **against zone coverage**, where routes with a hard break got the same receiver about **0.4 yd** more space.
- We report the limits as clearly as the wins. See [what didn't work](#honest-limitations).

## The event

The [NFL Big Data Bowl](https://operations.nfl.com/gameday/analytics/big-data-bowl/) is the NFL's annual sports-analytics competition. Teams get the league's player-tracking data (every player's position and speed, 10 times a second) and try to find new insights about the game. This project was built for the **NYU regional edition**. We used [its dataset](https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data), which covers weeks 1–8 of the 2021 season: 122 games and more than 34,000 receiver routes.

## The question

Scouts love top speed (40-yard dash times, GPS max speed). But a receiver gets open by **changing speed** at the right moment: exploding off the line (*gas*) and stopping hard at the break (*brakes*). Do those two skills tell us anything top speed doesn't?

## Approach

```
tracking data ──► per-route speed curves ──► per-player metrics ──► model ──► test on unseen weeks
 (01_build_routes)                          (02_player_metrics)   (03_models)  (06_ngs_check)
```

1. **Feature engineering.** Starting from raw x/y/speed frames, we cut out each route (snap to 3 s or the throw) and measured:
   - **Top speed:** how fast he runs on his fastest routes
   - **Burst (gas):** speed and acceleration in the first second
   - **Brake Score (brakes):** the biggest speed drop from his peak, on his hardest-breaking routes
2. **No peeking ahead.** Features come only from **weeks 1–4**. The targets come from **later weeks**: our own separation measure for weeks 5–8, and the NFL's official separation for weeks 9–18.
3. **Fair comparison.** Two models use the same algorithm (TabPFN, a pretrained tabular transformer) and the same splits. One gets *top speed only*; the other gets *top speed + gas + brakes*. Both are scored on 5×5 repeated cross-validation over **players**, so the model never sees the receiver it predicts.
4. **Sanity checks.** We reran everything with a simple linear model (RidgeCV), tried a different way of measuring separation, and ran a route-level test (12,001 routes) to see when braking matters.

## Results

| Test | Error, top speed only | Error, + gas & brakes | Improvement | Splits won |
|---|---|---|---|---|
| Our separation, weeks 5–8 (106 WRs) | 0.335 yd | 0.326 yd | 2.6% | 16 of 25 |
| **Official NFL Next Gen Stats, weeks 9–18 (67 WRs)** | **0.460 yd** | **0.418 yd** | **9.1%** | **21 of 25** |

*Error = mean absolute error in yards of separation (distance to the nearest defender).*

![Model comparison](outputs/figures/model_comparison.png)

**Brake Score leaders, weeks 1–4**

| # | Receiver | Brake Score (yd/s) | Top speed (yd/s) | Separation, wks 5–8 (yd) | vs. zone (yd) |
|---|---|---|---|---|---|
| 1 | Jarvis Landry | 4.48 | 7.73 | 2.71 | 2.95 |
| 2 | Jamison Crowder | 4.26 | 8.57 | 3.20 | 3.82 |
| 3 | Sterling Shepard | 3.85 | 7.84 | 3.21 | 3.40 |
| 4 | Darius Slayton | 3.84 | 8.43 | 2.61 | 3.32 |
| 5 | Laviska Shenault | 3.83 | 8.11 | 3.07 | 3.62 |
| | *Median of all 106 WRs* | *2.58* | *8.43* | *2.87* | *3.31* |

Most of the hardest brakers are not burners, and 7 of the top 10 beat the median separation against zone. Cooper Kupp, in his 2021 receiving triple-crown season, barely brakes at all: he wins at the break without stopping. Full top 10 and more in [docs/METHODS.md](docs/METHODS.md).

## Honest limitations

- **Small effect.** Speed metrics alone explain little of who gets open (cross-validated R² < 0 on our own measure). The best single predictor is how open a receiver got *earlier* in the season, so getting open is a lasting skill that speed only partly captures.
- **Role vs. skill.** The Brake Score correlates 0.90 with how many breaking routes (curls, comebacks, outs) a receiver is asked to run. The data has no route labels, so we can't fully separate talent from offensive role.
- **Model-dependent on the small test.** On our weeks 5–8 measure the gain shows up with TabPFN but not with a linear model. On the official NGS data it holds for both (9.1% and 8.1%).
- **Scale.** One season and about 100 receivers.

Every model variant we ran, including the ones that didn't work, is listed in [docs/METHODS.md](docs/METHODS.md).

## Who could use this

- **Scouts:** read a receiver's Brake Score next to his top speed and past separation when projecting who will get open.
- **Coaches:** against zone, routes with a hard break got the same receiver about 0.4 yd more space than his other routes.

## Repo structure

```
src/
  01_build_routes.py     raw tracking → one row per route + speed curves
  02_player_metrics.py   routes → per-player gas / brakes / top-speed metrics
  03_models.py           baseline vs. full model, 5×5 repeated CV
  04_charts.py           model comparison + metric vs. separation charts
  05_speed_curves.py     "same top speed, different brakes" chart
  06_ngs_check.py        out-of-sample test vs. official NFL Next Gen Stats
  07_break_routes.py     route-level check: when does braking matter?
outputs/figures/         charts used in this README
outputs/tables/          all results (metrics, predictions, model scores)
docs/METHODS.md          full methods, robustness checks, every model variant
docs/team_notes.md       working notes from the competition
```

## How to run

1. Download the event data from [nfl-big-data-bowl-regional-event-data](https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data) into `data/raw/` (so you have `data/raw/games.csv`, `data/raw/tracking/…`).
2. `pip install -r requirements.txt`. TabPFN needs a [Prior Labs](https://priorlabs.ai/) API key in the `TABPFN_TOKEN` environment variable. Without it, `03_models.py` falls back to RidgeCV.
3. From the repo root:
   ```bash
   python src/01_build_routes.py
   python src/02_player_metrics.py
   python src/03_models.py
   python src/04_charts.py
   python src/05_speed_curves.py
   python src/06_ngs_check.py     # optional: downloads public NGS data from nflverse
   python src/07_break_routes.py  # optional: route-level check, prints only
   ```
   If `06` fails with an SSL certificate error (common with python.org Python on macOS), download the file first:
   `curl -L -o data/raw/ngs_2021_receiving.csv.gz https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_2021_receiving.csv.gz`

## Data & credits

Tracking data: the NFL Big Data Bowl NYU regional event dataset (not redistributed here). Official separation: [nflverse](https://github.com/nflverse/nflverse-data) Next Gen Stats. Code is MIT-licensed.
